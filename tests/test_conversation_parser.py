"""Tests for ConversationParser - JSONL conversation file chunking."""

import pytest
from conductor_memory.search.parsers.conversation_parser import ConversationParser


@pytest.fixture
def parser():
    return ConversationParser()


class TestSupports:
    """Test file type detection."""

    def test_supports_jsonl(self, parser):
        assert parser.supports("session.jsonl") is True
        assert parser.supports("/path/to/conversation.jsonl") is True

    def test_rejects_other_formats(self, parser):
        assert parser.supports("file.json") is False
        assert parser.supports("script.py") is False
        assert parser.supports("data.csv") is False


class TestBasicParsing:
    """Test basic conversation parsing."""

    def test_simple_turn(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Hello"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi there! How can I help?"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        text, meta = chunks[0]
        assert "[User]: Hello" in text
        assert "[Claude]: Hi there!" in text
        assert meta.domain == "conversation"

    def test_multi_turn_conversation(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"What is Python?"}}
{"type":"assistant","message":{"role":"assistant","content":"Python is a programming language."}}
{"type":"user","message":{"role":"user","content":"How do I install it?"}}
{"type":"assistant","message":{"role":"assistant","content":"You can download it from python.org"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 2
        assert "What is Python?" in chunks[0][0]
        assert "How do I install it?" in chunks[1][0]

    def test_user_only_turn(self, parser):
        """User message without assistant response should still create chunk."""
        content = '''{"type":"user","message":{"role":"user","content":"Hello there"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        assert "[User]: Hello there" in chunks[0][0]
        assert "[Claude]" not in chunks[0][0]


class TestMetaMessageSkipping:
    """Test that meta and progress messages are skipped."""

    def test_skip_meta_messages(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Hello"},"isMeta":true}
{"type":"user","message":{"role":"user","content":"Real message"}}
{"type":"assistant","message":{"role":"assistant","content":"Response"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        assert "Hello" not in chunks[0][0]  # Meta message skipped
        assert "Real message" in chunks[0][0]

    def test_skip_progress_messages(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Hello"}}
{"type":"progress","data":{"type":"hook_progress"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        # Progress message shouldn't appear in output

    def test_skip_file_history_snapshot(self, parser):
        content = '''{"type":"file-history-snapshot","messageId":"abc"}
{"type":"user","message":{"role":"user","content":"Hello"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1


class TestToolCalls:
    """Test tool call handling."""

    def test_tool_use_summarized(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Read the file"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Let me read that."},{"type":"tool_use","name":"Read","input":{"file":"test.py"}}]}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        text = chunks[0][0]
        assert "Let me read that" in text
        assert "[Tool: Read]" in text
        assert "input" not in text  # Tool input not included

    def test_tool_result_summarized(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Show file"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_result","content":"file contents here"}]}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        assert "[Tool result]" in chunks[0][0]


class TestMetadataExtraction:
    """Test metadata extraction from JSONL."""

    def test_session_id_from_filename(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Hello"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi"}}'''

        chunks = parser.parse(content, "abc-123-def.jsonl")

        meta = chunks[0][1]
        assert "abc-123-def" in meta.module

    def test_project_from_cwd(self, parser):
        content = '''{"type":"user","cwd":"C:\\\\Users\\\\test\\\\projects\\\\myproject","message":{"role":"user","content":"Hello"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi"}}'''

        chunks = parser.parse(content, "test.jsonl")

        meta = chunks[0][1]
        assert "myproject" in meta.parent_class

    def test_project_from_unix_path(self, parser):
        content = '''{"type":"user","cwd":"/home/user/projects/myapp","message":{"role":"user","content":"Hello"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi"}}'''

        chunks = parser.parse(content, "test.jsonl")

        meta = chunks[0][1]
        assert "myapp" in meta.parent_class


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_json_line_skipped(self, parser):
        content = '''{"type":"user","message":{"role":"user","content":"Hello"}}
not valid json here
{"type":"assistant","message":{"role":"assistant","content":"Hi"}}'''

        # Should not raise, just skip bad line
        chunks = parser.parse(content, "test.jsonl")
        assert len(chunks) == 1

    def test_empty_content(self, parser):
        content = ""
        chunks = parser.parse(content, "test.jsonl")
        assert len(chunks) == 0

    def test_empty_content_skipped(self, parser):
        """Empty messages should be skipped."""
        content = '''{"type":"user","message":{"role":"user","content":""}}
{"type":"assistant","message":{"role":"assistant","content":""}}'''

        chunks = parser.parse(content, "test.jsonl")
        # Both messages empty, should result in no chunks
        assert len(chunks) == 0

    def test_consecutive_user_messages_grouped(self, parser):
        """Multiple user messages before assistant should group."""
        content = '''{"type":"user","message":{"role":"user","content":"First part of my question"}}
{"type":"user","message":{"role":"user","content":"And here is more context"}}
{"type":"assistant","message":{"role":"assistant","content":"I understand, here is my answer"}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        text = chunks[0][0]
        assert "First part" in text
        assert "more context" in text

    def test_consecutive_assistant_messages_grouped(self, parser):
        """Multiple assistant messages should group."""
        content = '''{"type":"user","message":{"role":"user","content":"Tell me everything about Python"}}
{"type":"assistant","message":{"role":"assistant","content":"Python is a high-level language."}}
{"type":"assistant","message":{"role":"assistant","content":"It was created by Guido van Rossum."}}'''

        chunks = parser.parse(content, "test.jsonl")

        assert len(chunks) == 1
        text = chunks[0][0]
        assert "high-level" in text
        assert "Guido" in text


class TestChunkingManagerIntegration:
    """Test integration with ChunkingManager."""

    def test_chunking_manager_uses_conversation_parser(self):
        from conductor_memory.search.chunking import ChunkingManager

        manager = ChunkingManager()
        content = '''{"type":"user","message":{"role":"user","content":"Hello world test message"}}
{"type":"assistant","message":{"role":"assistant","content":"Hi there, this is a response"}}'''

        chunks = manager.chunk_text(content, "conversation.jsonl")

        assert len(chunks) >= 1
        assert "Hello world" in chunks[0][0]
        assert chunks[0][1].domain == "conversation"
