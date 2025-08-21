from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, JSON, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from app.models.schemas import TestCaseStatus, TestCasePriority

Base = declarative_base()


class TestCaseModel(Base):
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=False)
    feature_description = Column(Text, nullable=False)
    acceptance_criteria = Column(Text, nullable=False)
    priority = Column(Enum(TestCasePriority), default=TestCasePriority.MEDIUM)
    status = Column(Enum(TestCaseStatus), default=TestCaseStatus.DRAFT)
    tags = Column(JSON, default=list)
    preconditions = Column(Text, nullable=True)
    test_steps = Column(JSON, nullable=False)
    expected_result = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(100), nullable=True)
    jira_issue_key = Column(String(50), nullable=True, index=True)
    zephyr_test_id = Column(String(100), nullable=True, index=True)
    # Store embedding as binary blob for compact storage and fast retrieval
    embedding_vector = Column(LargeBinary, nullable=True)

    def __repr__(self):
        return f"<TestCase(id={self.id}, title='{self.title}', status='{self.status}')>"
