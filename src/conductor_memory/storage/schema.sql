-- Conductor Memory PostgreSQL Schema
-- Fast metadata storage for dashboard operations

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS conductor;

-- Set search path for this session
SET search_path TO conductor, public;

-- Codebases table
CREATE TABLE IF NOT EXISTS codebases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    path TEXT NOT NULL,
    description TEXT,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexed files table
CREATE TABLE IF NOT EXISTS indexed_files (
    id SERIAL PRIMARY KEY,
    codebase_id INTEGER REFERENCES codebases(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,           -- Absolute path
    relative_path TEXT NOT NULL,       -- Relative to codebase root
    content_hash VARCHAR(64),
    file_size INTEGER,
    line_count INTEGER,
    language VARCHAR(50),
    indexed_at TIMESTAMP DEFAULT NOW(),
    modified_at TIMESTAMP,
    UNIQUE(codebase_id, relative_path)
);

-- Summaries table
CREATE TABLE IF NOT EXISTS summaries (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES indexed_files(id) ON DELETE CASCADE UNIQUE,
    summary_text TEXT,
    summary_type VARCHAR(20) DEFAULT 'llm',  -- 'simple' or 'llm'
    pattern VARCHAR(255),                     -- 'service', 'model', 'util', etc.
    domain VARCHAR(255),                      -- 'api', 'db', 'ui', etc.
    key_functions TEXT[],                     -- Array of function names
    dependencies TEXT[],                      -- Array of imports
    validation_status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
    validated_by VARCHAR(100),
    validated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_files_codebase ON indexed_files(codebase_id);
CREATE INDEX IF NOT EXISTS idx_files_relative_path ON indexed_files(relative_path);
CREATE INDEX IF NOT EXISTS idx_files_hash ON indexed_files(content_hash);
CREATE INDEX IF NOT EXISTS idx_summaries_file ON summaries(file_id);
CREATE INDEX IF NOT EXISTS idx_summaries_pattern ON summaries(pattern);
CREATE INDEX IF NOT EXISTS idx_summaries_domain ON summaries(domain);
CREATE INDEX IF NOT EXISTS idx_summaries_status ON summaries(validation_status);
CREATE INDEX IF NOT EXISTS idx_summaries_type ON summaries(summary_type);

-- Materialized view for dashboard stats (instant counts)
DROP MATERIALIZED VIEW IF EXISTS codebase_stats;
CREATE MATERIALIZED VIEW codebase_stats AS
SELECT
    c.id as codebase_id,
    c.name as codebase_name,
    COUNT(DISTINCT f.id) as indexed_count,
    COUNT(DISTINCT s.id) as summarized_count,
    COUNT(DISTINCT CASE WHEN s.summary_type = 'llm' THEN s.id END) as llm_count,
    COUNT(DISTINCT CASE WHEN s.summary_type = 'simple' THEN s.id END) as simple_count,
    COUNT(DISTINCT CASE WHEN s.validation_status = 'approved' THEN s.id END) as approved_count,
    COUNT(DISTINCT CASE WHEN s.validation_status = 'rejected' THEN s.id END) as rejected_count,
    COUNT(DISTINCT CASE WHEN s.validation_status = 'pending' THEN s.id END) as pending_count
FROM codebases c
LEFT JOIN indexed_files f ON f.codebase_id = c.id
LEFT JOIN summaries s ON s.file_id = f.id
GROUP BY c.id, c.name;

CREATE UNIQUE INDEX IF NOT EXISTS idx_stats_codebase ON codebase_stats(codebase_id);

-- Function to refresh stats (call after bulk operations)
CREATE OR REPLACE FUNCTION refresh_codebase_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY codebase_stats;
END;
$$ LANGUAGE plpgsql;
