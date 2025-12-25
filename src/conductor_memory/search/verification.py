"""
Verification search mode dataclasses for conductor-memory.

Phase 3 of the Implementation Context Enhancement Plan.
Provides structured types for "does X use pattern Y?" verification queries.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


# Pattern for matching method/class names including:
# - Simple names: _generate_features, process, MyClass
# - Qualified names: ClassName.method, module.Class.method
# - Names with underscores: __init__, _private_method
NAME_PATTERN = r"([\w_]+(?:\.[\w_]+)*)"

# Verification query patterns
# Each pattern captures (subject, claim) groups
# Ordered from most specific to least specific
VERIFY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "verify X uses Y" / "verify X use Y"
    (re.compile(rf"verify\s+{NAME_PATTERN}\s+uses?\s+(.+)", re.IGNORECASE), "verify_uses"),
    # "does X use Y"
    (re.compile(rf"does\s+{NAME_PATTERN}\s+use\s+(.+)", re.IGNORECASE), "does_use"),
    # "is X using Y"
    (re.compile(rf"is\s+{NAME_PATTERN}\s+using\s+(.+)", re.IGNORECASE), "is_using"),
    # "does X call Y"
    (re.compile(rf"does\s+{NAME_PATTERN}\s+call\s+(.+)", re.IGNORECASE), "does_call"),
    # "does X access Y"
    (re.compile(rf"does\s+{NAME_PATTERN}\s+access\s+(.+)", re.IGNORECASE), "does_access"),
    # "does X have Y"
    (re.compile(rf"does\s+{NAME_PATTERN}\s+have\s+(.+)", re.IGNORECASE), "does_have"),
    # "find if X uses Y"
    (re.compile(rf"find\s+if\s+{NAME_PATTERN}\s+uses?\s+(.+)", re.IGNORECASE), "find_if_uses"),
    # "check if X uses/calls/has/accesses Y"
    (re.compile(rf"check\s+if\s+{NAME_PATTERN}\s+(.+)", re.IGNORECASE), "check_if"),
    # "confirm X uses/calls/has Y"
    (re.compile(rf"confirm\s+{NAME_PATTERN}\s+(.+)", re.IGNORECASE), "confirm"),
]

# Stop words to filter out when extracting key terms
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "for", "of", "in", "on", "at", "by", "with",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "also", "now", "if", "or",
    "and", "but", "because", "until", "while", "that", "this", "these",
    "those", "what", "which", "who", "whom", "any", "both", "either",
    "window", "relative",  # Common but not helpful for code search
})


def parse_verification_query(query: str) -> Optional["VerificationIntent"]:
    """
    Parse a verification query to extract subject and claim.
    
    Handles queries like:
    - "verify _generate_features uses iloc"
    - "does MyClass.process call validate"
    - "is DataProcessor using caching"
    - "check if process_data accesses database"
    - "find if handler uses async"
    
    Args:
        query: The verification query string
        
    Returns:
        VerificationIntent with subject and claim, or None if not a verification query
        
    Examples:
        >>> parse_verification_query("verify _generate_features uses iloc")
        VerificationIntent(subject="_generate_features", claim="iloc")
        
        >>> parse_verification_query("does MyClass.process call validate")
        VerificationIntent(subject="MyClass.process", claim="validate")
        
        >>> parse_verification_query("what is the purpose of this class")
        None
    """
    if not query or not isinstance(query, str):
        return None
    
    # Normalize whitespace (collapse multiple spaces, strip edges)
    normalized = " ".join(query.split())
    
    for pattern, _pattern_type in VERIFY_PATTERNS:
        match = pattern.match(normalized)
        if match:
            subject = match.group(1).strip()
            claim = match.group(2).strip()
            
            # Skip if subject or claim is empty after stripping
            if not subject or not claim:
                continue
            
            return VerificationIntent(subject=subject, claim=claim)
    
    return None


def extract_key_terms(claim: str) -> List[str]:
    """
    Extract searchable key terms from a verification claim.
    
    Filters out common stop words and returns terms useful for searching
    implementation signals.
    
    Args:
        claim: The claim portion of a verification query
        
    Returns:
        List of key terms suitable for searching
        
    Examples:
        >>> extract_key_terms("window-relative bar_index for DataFrame access")
        ['bar_index', 'DataFrame', 'access']
        
        >>> extract_key_terms("uses iloc with sliding window")
        ['iloc', 'sliding']
        
        >>> extract_key_terms("the repository pattern")
        ['repository', 'pattern']
    """
    if not claim or not isinstance(claim, str):
        return []
    
    # Split on whitespace and common separators, preserving underscores
    # Handle hyphenated terms by splitting them
    terms = re.split(r"[\s,;:\-\(\)\[\]\{\}]+", claim)
    
    # Filter and clean terms
    key_terms = []
    for term in terms:
        # Strip any remaining punctuation from edges
        cleaned = term.strip(".,!?\"'`")
        
        # Skip empty terms, single characters (except meaningful ones like 'x'),
        # and stop words
        if not cleaned:
            continue
        if len(cleaned) == 1 and cleaned.lower() not in {'x', 'y', 'n', 'i', 'j', 'k'}:
            continue
        if cleaned.lower() in STOP_WORDS:
            continue
            
        key_terms.append(cleaned)
    
    return key_terms


def is_verification_query(query: str) -> bool:
    """
    Check if a query is a verification query without fully parsing it.
    
    Useful for quick filtering before more expensive operations.
    
    Args:
        query: The query string to check
        
    Returns:
        True if the query appears to be a verification query
    """
    if not query or not isinstance(query, str):
        return False
    
    normalized = query.lower().strip()
    
    # Quick prefix checks for common verification patterns
    verification_prefixes = (
        "verify ", "does ", "is ", "check if ", "confirm ", "find if ",
    )
    
    return normalized.startswith(verification_prefixes)


class VerificationStatus(str, Enum):
    """
    Status of a verification query result.
    
    SUPPORTED: Evidence found supporting the claim
    NOT_SUPPORTED: Subject found but no evidence for claim
    CONTRADICTED: Evidence suggests opposite of claim
    INCONCLUSIVE: Some evidence but not definitive
    SUBJECT_NOT_FOUND: Could not locate the subject
    """
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    SUBJECT_NOT_FOUND = "subject_not_found"


@dataclass
class VerificationIntent:
    """
    Parsed intent from a verification query.
    
    Extracted by the query parser from queries like:
    - "verify _generate_features uses window-relative bar_index"
    - "does MyClass use the repository pattern?"
    - "check if process_data calls validate"
    
    Attributes:
        subject: The method/class/file being verified (e.g., "_generate_features")
        claim: What we're verifying about it (e.g., "uses window-relative bar_index")
    """
    subject: str
    claim: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subject": self.subject,
            "claim": self.claim,
        }


@dataclass
class Evidence:
    """
    A piece of evidence supporting or contradicting a verification claim.
    
    Found by matching implementation signals against the claim.
    
    Attributes:
        type: Type of evidence (e.g., "subscript_access", "call", "attribute_read")
        detail: The actual signal found (e.g., "df.iloc[bar_index]")
        relevance: How relevant this evidence is to the claim (0.0 to 1.0)
        line: Line number where evidence was found, if available
    """
    type: str
    detail: str
    relevance: float
    line: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "detail": self.detail,
            "relevance": self.relevance,
        }
        if self.line is not None:
            result["line"] = self.line
        return result


@dataclass
class SubjectInfo:
    """
    Information about the subject being verified.
    
    Attributes:
        name: Name of the method/class/file (e.g., "_generate_features")
        file: File path where subject was found
        found: Whether the subject was located in the codebase
        line: Line number of subject definition, if available
        type: Type of subject (e.g., "method", "class", "file")
    """
    name: str
    file: Optional[str] = None
    found: bool = False
    line: Optional[int] = None
    type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "name": self.name,
            "found": self.found,
        }
        if self.file is not None:
            result["file"] = self.file
        if self.line is not None:
            result["line"] = self.line
        if self.type is not None:
            result["type"] = self.type
        return result


@dataclass
class VerificationInfo:
    """
    Verification status and supporting evidence.
    
    Attributes:
        status: The verification result status
        confidence: Confidence in the verification (0.0 to 1.0)
        evidence: List of evidence items found
    """
    status: VerificationStatus
    confidence: float
    evidence: List[Evidence] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class VerificationResult:
    """
    Complete result of a verification search query.
    
    Returned by search_async() when search_mode="verify".
    
    Attributes:
        search_mode: Always "verify" for verification results
        subject: Information about the subject being verified
        verification: The verification status and evidence
        summary: Human-readable summary of the verification result
    """
    subject: SubjectInfo
    verification: VerificationInfo
    summary: str
    search_mode: str = "verify"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "search_mode": self.search_mode,
            "subject": self.subject.to_dict(),
            "verification": self.verification.to_dict(),
            "summary": self.summary,
        }
    
    @classmethod
    def subject_not_found(cls, subject_name: str, claim: str) -> "VerificationResult":
        """
        Factory method for creating a result when subject cannot be found.
        
        Args:
            subject_name: Name of the subject that wasn't found
            claim: The claim that was being verified
            
        Returns:
            VerificationResult with SUBJECT_NOT_FOUND status
        """
        return cls(
            subject=SubjectInfo(name=subject_name, found=False),
            verification=VerificationInfo(
                status=VerificationStatus.SUBJECT_NOT_FOUND,
                confidence=1.0,
                evidence=[],
            ),
            summary=f"Could not locate '{subject_name}' in the codebase.",
        )
    
    @classmethod
    def not_supported(
        cls, 
        subject_name: str, 
        file: str, 
        claim: str,
        line: Optional[int] = None,
        subject_type: Optional[str] = None,
    ) -> "VerificationResult":
        """
        Factory method for creating a result when no evidence supports the claim.
        
        Args:
            subject_name: Name of the subject
            file: File where subject was found
            claim: The claim that was being verified
            line: Line number of subject, if available
            subject_type: Type of subject (method, class, etc.)
            
        Returns:
            VerificationResult with NOT_SUPPORTED status
        """
        return cls(
            subject=SubjectInfo(
                name=subject_name, 
                file=file, 
                found=True, 
                line=line,
                type=subject_type,
            ),
            verification=VerificationInfo(
                status=VerificationStatus.NOT_SUPPORTED,
                confidence=0.8,
                evidence=[],
            ),
            summary=f"Found '{subject_name}' but no evidence that it {claim}.",
        )


# =============================================================================
# Evidence Matching Functions (Phase 3.3)
# =============================================================================

# Mapping from tag prefixes to evidence types
TAG_PREFIX_TO_EVIDENCE_TYPE: Dict[str, str] = {
    "calls:": "call",
    "reads:": "attribute_read",
    "writes:": "attribute_write",
    "subscript:": "subscript_access",
    "param:": "parameter_usage",
}

# Patterns for extracting signals from [Implementation Signals] section in content
CONTENT_SIGNAL_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"Calls:\s*(.+)$", re.MULTILINE), "call"),
    (re.compile(r"Reads:\s*(.+)$", re.MULTILINE), "attribute_read"),
    (re.compile(r"Writes:\s*(.+)$", re.MULTILINE), "attribute_write"),
    (re.compile(r"Subscripts:\s*(.+)$", re.MULTILINE), "subscript_access"),
    (re.compile(r"Parameters used:\s*(.+)$", re.MULTILINE), "parameter_usage"),
]


def matches_any(signal: str, terms: List[str]) -> bool:
    """
    Check if a signal matches any of the given terms.
    
    Matching is case-insensitive and supports:
    - Exact match: term equals signal (case-insensitive)
    - Substring match: term is contained in signal
    - Token match: term appears as a word boundary in signal
    
    Args:
        signal: The implementation signal to check (e.g., "df.iloc[bar_index]")
        terms: List of key terms extracted from the claim
        
    Returns:
        True if the signal matches at least one term
        
    Examples:
        >>> matches_any("df.iloc[bar_index]", ["iloc", "bar_index"])
        True
        
        >>> matches_any("self._cache", ["cache"])
        True
        
        >>> matches_any("process_data", ["analyze"])
        False
    """
    if not signal or not terms:
        return False
    
    signal_lower = signal.lower()
    
    for term in terms:
        if not term:
            continue
        term_lower = term.lower()
        
        # Exact match (ignoring case)
        if signal_lower == term_lower:
            return True
        
        # Substring match - term appears somewhere in signal
        if term_lower in signal_lower:
            return True
        
        # Also check if signal appears in term (for abbreviated tags)
        if signal_lower in term_lower:
            return True
    
    return False


def calculate_relevance(signal: str, claim_terms: List[str]) -> float:
    """
    Calculate relevance score for a signal against claim terms.
    
    Scoring approach:
    - Base score: 0.5 for any match
    - Exact match bonus: +0.4 (signal equals a term exactly)
    - Multi-term bonus: +0.1 per additional matching term (max +0.3)
    - Longer match bonus: +0.1 for terms that are substantial (4+ chars)
    
    Maximum score is clamped to 1.0.
    
    Args:
        signal: The implementation signal (e.g., "df.iloc[bar_index]")
        claim_terms: Key terms from the verification claim
        
    Returns:
        Relevance score from 0.0 to 1.0
        
    Examples:
        >>> calculate_relevance("iloc", ["iloc"])  # Exact match
        0.95
        
        >>> calculate_relevance("df.iloc[bar_index]", ["iloc", "bar_index"])
        0.95  # Multiple terms match
        
        >>> calculate_relevance("self._cache", ["cache"])
        0.6  # Substring match
    """
    if not signal or not claim_terms:
        return 0.0
    
    signal_lower = signal.lower()
    matching_terms = []
    has_exact_match = False
    
    for term in claim_terms:
        if not term:
            continue
        term_lower = term.lower()
        
        # Check for exact match
        if signal_lower == term_lower:
            has_exact_match = True
            matching_terms.append(term)
        # Check for substring match
        elif term_lower in signal_lower or signal_lower in term_lower:
            matching_terms.append(term)
    
    if not matching_terms:
        return 0.0
    
    # Base score for any match
    score = 0.5
    
    # Exact match bonus
    if has_exact_match:
        score += 0.4
    
    # Multi-term bonus: additional matching terms beyond the first
    additional_matches = len(matching_terms) - 1
    score += min(additional_matches * 0.1, 0.3)
    
    # Longer term bonus: reward for matching substantial terms
    substantial_matches = sum(1 for t in matching_terms if len(t) >= 4)
    if substantial_matches > 0:
        score += 0.05
    
    return min(score, 1.0)


def _extract_signals_from_tags(tags: List[str]) -> List[Tuple[str, str, str]]:
    """
    Extract signal information from chunk tags.
    
    Args:
        tags: List of tags (e.g., ["calls:iloc", "subscript:iloc", "param:bar_index"])
        
    Returns:
        List of tuples: (evidence_type, signal_value, original_tag)
    """
    signals = []
    
    for tag in tags:
        for prefix, evidence_type in TAG_PREFIX_TO_EVIDENCE_TYPE.items():
            if tag.startswith(prefix):
                signal_value = tag[len(prefix):]
                signals.append((evidence_type, signal_value, tag))
                break
    
    return signals


def _extract_signals_from_content(content: str) -> List[Tuple[str, str]]:
    """
    Extract signals from [Implementation Signals] section in chunk content.
    
    Args:
        content: The chunk content which may contain [Implementation Signals] section
        
    Returns:
        List of tuples: (evidence_type, signal_detail)
    """
    signals = []
    
    # Check if content has implementation signals section
    if "[Implementation Signals]" not in content:
        return signals
    
    for pattern, evidence_type in CONTENT_SIGNAL_PATTERNS:
        match = pattern.search(content)
        if match:
            # Split the comma-separated values
            values = match.group(1).strip()
            for value in values.split(","):
                value = value.strip()
                if value:
                    signals.append((evidence_type, value))
    
    return signals


def find_evidence(
    chunk_tags: List[str], 
    chunk_content: str, 
    claim: str
) -> List[Evidence]:
    """
    Find evidence in a chunk's tags and content that supports a verification claim.
    
    This function searches for implementation signals that match terms from the claim.
    It extracts evidence from two sources:
    1. Chunk tags (e.g., "calls:iloc", "subscript:iloc", "reads:self._cache")
    2. Chunk content's [Implementation Signals] section
    
    Evidence types:
    - "call": Method/function calls (from calls:* tags)
    - "attribute_read": Attribute reads (from reads:* tags)
    - "attribute_write": Attribute writes (from writes:* tags)
    - "subscript_access": Subscript/index patterns (from subscript:* tags)
    - "parameter_usage": Parameter usage (from param:* tags)
    
    Args:
        chunk_tags: List of tags from the chunk metadata
        chunk_content: The chunk's text content (may include [Implementation Signals] section)
        claim: The claim portion of the verification query
        
    Returns:
        List of Evidence objects sorted by relevance (highest first)
        
    Examples:
        >>> find_evidence(
        ...     tags=["calls:iloc", "subscript:iloc"],
        ...     content="...",
        ...     claim="uses iloc for indexing"
        ... )
        [Evidence(type="subscript_access", detail="iloc", relevance=0.95),
         Evidence(type="call", detail="iloc", relevance=0.95)]
         
        >>> find_evidence(
        ...     tags=["param:bar_index", "reads:self._df"],
        ...     content="...[Implementation Signals]\\nSubscripts: df.iloc[bar_index]",
        ...     claim="uses bar_index parameter for DataFrame access"
        ... )
        [Evidence(type="subscript_access", detail="df.iloc[bar_index]", relevance=0.85),
         Evidence(type="parameter_usage", detail="bar_index", relevance=0.95)]
    """
    evidence: List[Evidence] = []
    seen_signals: set = set()  # Avoid duplicates
    
    # Extract key terms from the claim
    claim_terms = extract_key_terms(claim)
    
    if not claim_terms:
        return evidence
    
    # Extract evidence from tags
    tag_signals = _extract_signals_from_tags(chunk_tags or [])
    for evidence_type, signal_value, _original_tag in tag_signals:
        if matches_any(signal_value, claim_terms):
            # Create a unique key to avoid duplicates
            key = (evidence_type, signal_value)
            if key not in seen_signals:
                seen_signals.add(key)
                relevance = calculate_relevance(signal_value, claim_terms)
                evidence.append(Evidence(
                    type=evidence_type,
                    detail=signal_value,
                    relevance=relevance,
                ))
    
    # Extract evidence from content (Implementation Signals section)
    content_signals = _extract_signals_from_content(chunk_content or "")
    for evidence_type, signal_detail in content_signals:
        if matches_any(signal_detail, claim_terms):
            # Create a unique key - content signals may be more detailed
            # but we don't want exact duplicates
            key = (evidence_type, signal_detail)
            if key not in seen_signals:
                seen_signals.add(key)
                relevance = calculate_relevance(signal_detail, claim_terms)
                evidence.append(Evidence(
                    type=evidence_type,
                    detail=signal_detail,
                    relevance=relevance,
                ))
    
    # Sort by relevance (highest first), with detail as tiebreaker for deterministic ordering
    evidence.sort(key=lambda e: (e.relevance, e.detail), reverse=True)
    
    return evidence
