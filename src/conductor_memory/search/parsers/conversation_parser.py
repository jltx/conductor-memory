"""
Parser for Claude Code JSONL conversation files.

Provides semantic chunking at message boundaries, grouping user messages
with their corresponding assistant responses for coherent retrieval.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .base import ContentParser
from ..chunking import ChunkMetadata

logger = logging.getLogger(__name__)


class ConversationParser(ContentParser):
    """Parser for Claude Code JSONL conversation files.

    Creates chunks at conversation turn boundaries (user + assistant pair),
    preserving session and project metadata for filtering.
    """

    # Message types to include in chunks
    CONTENT_TYPES = {'user', 'assistant'}

    # Message types to skip entirely
    SKIP_TYPES = {'progress', 'file-history-snapshot'}

    def supports(self, file_path: str) -> bool:
        """Check if this parser supports the given file."""
        return file_path.endswith('.jsonl')

    def parse(
        self,
        content: str,
        file_path: str,
        commit_hash: Optional[str] = None
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Parse JSONL conversation into chunks.

        Groups user messages with their assistant responses into turns.
        Extracts session_id from filename and project from cwd field.

        Args:
            content: Raw JSONL content
            file_path: Path to the conversation file
            commit_hash: Optional git commit (unused for conversations)

        Returns:
            List of (chunk_text, metadata) tuples
        """
        session_id = self._extract_session_id(file_path)
        lines = content.strip().split('\n')

        # Parse all messages
        messages = []
        project = None

        for line_num, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed JSON at line {line_num}")
                continue

            # Skip meta messages
            if data.get('isMeta'):
                continue

            msg_type = data.get('type')

            # Skip non-content types
            if msg_type in self.SKIP_TYPES:
                continue

            # Extract project from cwd if available
            if not project and 'cwd' in data:
                project = self._extract_project_name(data['cwd'])

            # Only process content messages
            if msg_type not in self.CONTENT_TYPES:
                continue

            message_content = self._extract_message_content(data)
            if not message_content:
                continue

            messages.append({
                'type': msg_type,
                'content': message_content,
                'line_num': line_num,
                'timestamp': data.get('timestamp')
            })

        # Group into turns (user + assistant pairs)
        chunks = self._group_into_turns(
            messages,
            file_path,
            session_id,
            project or 'unknown'
        )

        return chunks

    def _extract_session_id(self, file_path: str) -> str:
        """Extract session ID from filename.

        Expected format: <session-uuid>.jsonl
        """
        path = Path(file_path)
        return path.stem  # Filename without extension

    def _extract_project_name(self, cwd: str) -> str:
        """Extract project name from working directory path.

        Handles both Windows and Unix paths.
        """
        # Normalize path separators
        cwd = cwd.replace('\\', '/')

        # Get the last path component (project folder)
        parts = [p for p in cwd.split('/') if p]
        if parts:
            return parts[-1]
        return 'unknown'

    def _extract_message_content(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a message object.

        Handles both string content and structured content with tool calls.
        """
        message = data.get('message', {})
        content = message.get('content')

        if content is None:
            return None

        # Simple string content
        if isinstance(content, str):
            return self._clean_content(content)

        # Structured content (array of blocks)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
                        text_parts.append(block.get('text', ''))
                    elif block_type == 'tool_use':
                        # Summarize tool calls
                        tool_name = block.get('name', 'unknown')
                        text_parts.append(f"[Tool: {tool_name}]")
                    elif block_type == 'tool_result':
                        # Summarize tool results briefly
                        text_parts.append("[Tool result]")

            return self._clean_content('\n'.join(text_parts))

        return None

    def _clean_content(self, content: str) -> str:
        """Clean and normalize message content."""
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()

        # Skip if empty (but allow short messages like "Hello")
        if len(content) < 2:
            return ''

        return content

    def _group_into_turns(
        self,
        messages: List[Dict[str, Any]],
        file_path: str,
        session_id: str,
        project: str
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Group messages into user+assistant turn pairs.

        Creates one chunk per turn. Handles consecutive messages of
        the same type by grouping them.
        """
        chunks = []
        turn_index = 0
        i = 0

        while i < len(messages):
            msg = messages[i]

            # Look for user message to start a turn
            if msg['type'] == 'user':
                user_content = msg['content']
                start_line = msg['line_num']
                end_line = start_line
                timestamp = msg.get('timestamp')

                # Collect any consecutive user messages
                while i + 1 < len(messages) and messages[i + 1]['type'] == 'user':
                    i += 1
                    user_content += '\n\n' + messages[i]['content']
                    end_line = messages[i]['line_num']

                # Look for assistant response
                assistant_content = ''
                if i + 1 < len(messages) and messages[i + 1]['type'] == 'assistant':
                    i += 1
                    assistant_content = messages[i]['content']
                    end_line = messages[i]['line_num']

                    # Collect consecutive assistant messages
                    while i + 1 < len(messages) and messages[i + 1]['type'] == 'assistant':
                        i += 1
                        assistant_content += '\n\n' + messages[i]['content']
                        end_line = messages[i]['line_num']

                # Build chunk text
                chunk_text = f"[User]: {user_content}"
                if assistant_content:
                    chunk_text += f"\n\n[Claude]: {assistant_content}"

                # Create metadata
                metadata = ChunkMetadata(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    commit_hash=None,
                    token_count=len(chunk_text.split()),
                    domain='conversation',
                    module=f"session:{session_id}",
                    parent_class=f"project:{project}"
                )

                chunks.append((chunk_text, metadata))
                turn_index += 1

            i += 1

        return chunks
