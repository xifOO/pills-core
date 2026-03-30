_DOMAIN_TAG_FIELDS: frozenset[str] = frozenset(
    {"is_ratio", "is_monetary", "is_rate", "is_score", "is_count"}
)


class AnalyzerError(Exception):
    """Base exception for all analyzer errors."""


class AmbiguousAnalyzerError(AnalyzerError):
    """Raised when more than one analyzer can handle a column."""

    def __init__(self, column_name: str, analyzer_names: list[str]) -> None:
        self.column_name = column_name
        self.analyzer_names = analyzer_names
        super().__init__(
            f"Ambiguous analyzers for column '{column_name}': "
            f"{', '.join(analyzer_names)}"
        )


class NoAnalyzerFoundError(AnalyzerError):
    """Raised when no registered analyzer can handle a column."""

    def __init__(self, column_name: str) -> None:
        self.column_name = column_name
        super().__init__(f"No analyzer registered for column '{column_name}'")


class UnknownDomainTagError(AnalyzerError):
    """Raised when a domain rule references a tag field that does not exist."""

    def __init__(self, tag_name: str) -> None:
        self.tag_name = tag_name
        super().__init__(
            f"Unknown domain tag '{tag_name}'. Valid tags: {list(_DOMAIN_TAG_FIELDS)}"
        )
