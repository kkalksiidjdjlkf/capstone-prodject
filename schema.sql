-- PostgreSQL Schema for KREPS RAG System
-- Hybrid Database: PostgreSQL (structured data) + FAISS (vectors)
-- Based on detailed specification with Korean language support

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ===========================================
-- CORE TABLES
-- ===========================================

-- Documents Table: Central registry for all uploaded documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(10) CHECK (file_type IN ('pdf', 'txt', 'md', 'csv')),
    language VARCHAR(10) CHECK (language IN ('ko', 'en', 'mixed')),
    sensitivity_level VARCHAR(15) CHECK (sensitivity_level IN ('public', 'internal', 'confidential')),
    chunk_count INTEGER DEFAULT 0,
    is_indexed BOOLEAN DEFAULT FALSE,
    file_hash VARCHAR(64) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Document Chunks Table: Stores individual text segments
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL CHECK (LENGTH(chunk_text) <= 2000),
    chunk_index INTEGER NOT NULL,
    start_char INTEGER,
    end_char INTEGER,
    language VARCHAR(10) CHECK (language IN ('ko', 'en', 'mixed')),
    metadata_json JSONB,
    embedding_id VARCHAR(100), -- Links to FAISS vector
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Query Logs Table: Complete audit trail of all user interactions
CREATE TABLE query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID, -- Optional for now, can reference users table later
    original_query TEXT NOT NULL,
    detected_language VARCHAR(10),
    llm_response TEXT,
    processing_time_ms INTEGER,
    retrieved_chunk_count INTEGER,
    avg_similarity_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================================
-- SUPPORTING TABLES
-- ===========================================

-- Users Table: Basic user management (optional for prototype)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE,
    email VARCHAR(255),
    role VARCHAR(20) CHECK (role IN ('admin', 'user', 'viewer')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Audit Logs Table: System-wide event tracking
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    action VARCHAR(100),
    table_name VARCHAR(50),
    record_id UUID,
    old_values JSONB,
    new_values JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- System Metrics Table: Performance data collection
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10,2),
    metric_unit VARCHAR(20),
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- ===========================================
-- INDEXES FOR PERFORMANCE
-- ===========================================

-- Documents table indexes
CREATE INDEX idx_documents_file_type ON documents(file_type);
CREATE INDEX idx_documents_language ON documents(language);
CREATE INDEX idx_documents_sensitivity ON documents(sensitivity_level);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);

-- Document chunks indexes
CREATE INDEX idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_chunks_language ON document_chunks(language);
CREATE INDEX idx_chunks_embedding_id ON document_chunks(embedding_id);
CREATE INDEX idx_chunks_created_at ON document_chunks(created_at);

-- Query logs indexes
CREATE INDEX idx_query_logs_user_id ON query_logs(user_id);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at);
CREATE INDEX idx_query_logs_language ON query_logs(detected_language);
CREATE INDEX idx_query_logs_processing_time ON query_logs(processing_time_ms);

-- Composite indexes for common queries
CREATE INDEX idx_documents_language_sensitivity ON documents(language, sensitivity_level);
CREATE INDEX idx_chunks_document_language ON document_chunks(document_id, language);
CREATE INDEX idx_query_logs_user_date ON query_logs(user_id, created_at);

-- ===========================================
-- KOREAN LANGUAGE SUPPORT
-- ===========================================

-- Function to normalize Korean text (NFC normalization)
CREATE OR REPLACE FUNCTION normalize_korean_text(input_text TEXT)
RETURNS TEXT AS $$
BEGIN
    -- Apply NFC normalization for Korean text
    RETURN normalize(input_text, NFC);
END;
$$ LANGUAGE plpgsql;

-- Function to count Korean characters
CREATE OR REPLACE FUNCTION count_korean_chars(input_text TEXT)
RETURNS INTEGER AS $$
DECLARE
    korean_count INTEGER := 0;
    i INTEGER;
    char_code INTEGER;
BEGIN
    FOR i IN 1..length(input_text) LOOP
        char_code := ascii(substring(input_text, i, 1));
        -- Korean Unicode ranges: 0xAC00-0xD7A3 (Hangul Syllables)
        IF char_code >= 44032 AND char_code <= 55203 THEN
            korean_count := korean_count + 1;
        END IF;
    END LOOP;
    RETURN korean_count;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- DATA INTEGRITY & CONSTRAINTS
-- ===========================================

-- Update trigger for documents.updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- INITIAL DATA SEEDING
-- ===========================================

-- Insert default admin user (for development)
INSERT INTO users (username, email, role) VALUES
('admin', 'admin@kreps.com', 'admin')
ON CONFLICT (username) DO NOTHING;

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit) VALUES
('total_documents', 0, 'count'),
('total_chunks', 0, 'count'),
('total_queries', 0, 'count'),
('avg_query_time', 0, 'ms')
ON CONFLICT (metric_name) DO NOTHING;

-- ===========================================
-- USEFUL VIEWS
-- ===========================================

-- View for document statistics
CREATE VIEW document_stats AS
SELECT
    language,
    sensitivity_level,
    file_type,
    COUNT(*) as count,
    AVG(chunk_count) as avg_chunks_per_doc
FROM documents
GROUP BY language, sensitivity_level, file_type;

-- View for query analytics
CREATE VIEW query_analytics AS
SELECT
    DATE(created_at) as query_date,
    detected_language,
    COUNT(*) as query_count,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(avg_similarity_score) as avg_similarity
FROM query_logs
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at), detected_language
ORDER BY query_date DESC;

-- View for system health
CREATE VIEW system_health AS
SELECT
    metric_name,
    metric_value,
    metric_unit,
    recorded_at
FROM system_metrics
WHERE recorded_at >= CURRENT_DATE - INTERVAL '1 day'
ORDER BY recorded_at DESC;

-- ===========================================
-- SECURITY POLICIES (Row Level Security)
-- ===========================================

-- Enable RLS on sensitive tables
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_logs ENABLE ROW LEVEL SECURITY;

-- Policy for documents (users can only see documents they have access to)
-- Note: This is a basic implementation. In production, you'd have proper user-document relationships.

-- ===========================================
-- BACKUP & RECOVERY SETUP
-- ===========================================

-- Enable WAL archiving for point-in-time recovery
-- Note: In production, configure wal_level, archive_mode, archive_command

-- ===========================================
-- MONITORING & ALERTS
-- ===========================================

-- Function to update system metrics
CREATE OR REPLACE FUNCTION update_system_metrics()
RETURNS VOID AS $$
BEGIN
    -- Update document count
    UPDATE system_metrics
    SET metric_value = (SELECT COUNT(*) FROM documents),
        recorded_at = NOW()
    WHERE metric_name = 'total_documents';

    -- Update chunk count
    UPDATE system_metrics
    SET metric_value = (SELECT COUNT(*) FROM document_chunks),
        recorded_at = NOW()
    WHERE metric_name = 'total_chunks';

    -- Update query count
    UPDATE system_metrics
    SET metric_value = (SELECT COUNT(*) FROM query_logs),
        recorded_at = NOW()
    WHERE metric_name = 'total_queries';

    -- Update average query time
    UPDATE system_metrics
    SET metric_value = (
        SELECT COALESCE(AVG(processing_time_ms), 0)
        FROM query_logs
        WHERE created_at >= CURRENT_DATE - INTERVAL '1 day'
    ),
    recorded_at = NOW()
    WHERE metric_name = 'avg_query_time';
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- SAMPLE DATA FOR TESTING
-- ===========================================

-- Sample document (comment out in production)
-- INSERT INTO documents (file_name, file_path, file_type, language, sensitivity_level, chunk_count, is_indexed, file_hash)
-- VALUES ('sample.pdf', '/data/sample.pdf', 'pdf', 'ko', 'internal', 10, true, 'abc123...');

COMMENT ON TABLE documents IS 'Central registry for all uploaded documents with Korean language support';
COMMENT ON TABLE document_chunks IS 'Individual text segments with metadata and FAISS vector linkage';
COMMENT ON TABLE query_logs IS 'Complete audit trail of all user interactions and performance metrics';
COMMENT ON TABLE users IS 'User management for access control and sessions';
COMMENT ON TABLE audit_logs IS 'System-wide event tracking for compliance';
COMMENT ON TABLE system_metrics IS 'Performance data collection and monitoring';

-- ===========================================
-- FINAL NOTES
-- ===========================================
-- This schema provides:
-- 1. Complete data integrity with foreign keys and constraints
-- 2. Korean language support with normalization functions
-- 3. Comprehensive audit trails for compliance
-- 4. Performance optimization with proper indexing
-- 5. Scalability features for growing data volumes
-- 6. Integration points with FAISS vector database
