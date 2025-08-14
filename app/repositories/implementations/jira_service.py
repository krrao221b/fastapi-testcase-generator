import httpx
from typing import Optional, Dict, Any
import structlog
from app.repositories.interfaces.jira_service import IJiraService
from app.config.settings import settings

logger = structlog.get_logger()


class AtlassianJiraService(IJiraService):
    """Atlassian JIRA Cloud implementation of JIRA service"""
    
    def __init__(self):
        self.base_url = settings.jira_base_url
        self.username = settings.jira_username
        self.api_token = settings.jira_api_token
        self.auth = (self.username, self.api_token) if self.username and self.api_token else None
    
    async def create_issue(self, title: str, description: str, issue_type: str = "Story") -> Optional[str]:
        """Create a JIRA issue and return the issue key"""
        if not self._is_configured():
            logger.warning("JIRA service not configured")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "fields": {
                        "project": {"key": "TC"},  # Default project key - should be configurable
                        "summary": title,
                        "description": description,
                        "issuetype": {"name": issue_type}
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/rest/api/3/issue",
                    json=payload,
                    auth=self.auth,
                    headers={"Accept": "application/json", "Content-Type": "application/json"}
                )
                
                if response.status_code == 201:
                    issue_data = response.json()
                    issue_key = issue_data.get("key")
                    logger.info("JIRA issue created successfully", issue_key=issue_key)
                    return issue_key
                else:
                    logger.error("Failed to create JIRA issue", 
                               status_code=response.status_code, 
                               response=response.text)
                    return None
                    
        except Exception as e:
            logger.error("Error creating JIRA issue", error=str(e))
            return None
    
    async def get_issue(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """Get JIRA issue details and return a simplified dict for frontend."""
        if not self._is_configured():
            logger.warning("JIRA service not configured")
            return None
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/rest/api/3/issue/{issue_key}",
                    auth=self.auth,
                    headers={"Accept": "application/json"}
                )
                if response.status_code == 200:
                    issue_data = response.json()
                    fields = issue_data.get("fields", {})
                    # Extract plain text from Atlassian Document Format
                    def extract_plain_text(desc):
                        if isinstance(desc, dict) and "content" in desc:
                            text_parts = []
                            def extract(node):
                                if isinstance(node, dict):
                                    if node.get("type") == "text":
                                        text_parts.append(node.get("text", ""))
                                    elif "content" in node:
                                        for child in node["content"]:
                                            extract(child)
                                elif isinstance(node, list):
                                    for item in node:
                                        extract(item)
                            extract(desc["content"])
                            return " ".join(text_parts)
                        elif isinstance(desc, str):
                            return desc
                        return ""
                    description2 = extract_plain_text(fields.get("description"))
                    summary = fields.get("summary", "")
                    status = fields.get("status", {}).get("name", "")
                    priority = fields.get("priority", {}).get("name", "")
                    assignee = fields.get("assignee", {}).get("displayName", "") if fields.get("assignee") else ""
                    attachments = [
                        {
                            "filename": att.get("filename"),
                            "url": att.get("content")
                        }
                        for att in fields.get("attachment", [])
                    ]
                    import re
                    split = re.split(r'Acceptance Criteria[:\n]+', description2, flags=re.IGNORECASE)
                    description = split[0].strip()
                    acceptance_criteria = split[1].strip() if len(split) > 1 else ""
                    acceptance_criteria = re.sub(r'!\S+?\.(jpg|png|jpeg|gif)[^!]*!', '', acceptance_criteria, flags=re.IGNORECASE).strip()
                    acceptance_criteria = re.sub(r'\[.*?\|.*?\]', '', acceptance_criteria)
                    acceptance_criteria = "\n".join([line for line in acceptance_criteria.splitlines() if line.strip()])
                    if not acceptance_criteria:
                        custom_ac_field = fields.get("customfield_10000")
                        if custom_ac_field:
                            acceptance_criteria = str(custom_ac_field).strip()
                    return {
                        "key": issue_data.get("key"),
                        "summary": summary,
                        "description": description,
                        "acceptance_criteria": acceptance_criteria,
                        "status": status,
                        "priority": priority,
                        "assignee": assignee,
                        "attachments": attachments,
                    }
                else:
                    logger.error("Failed to get JIRA issue", 
                               issue_key=issue_key,
                               status_code=response.status_code)
                    return None
        except Exception as e:
            logger.error("Error getting JIRA issue", issue_key=issue_key, error=str(e))
            return None
    
    async def update_issue(self, issue_key: str, fields: Dict[str, Any]) -> bool:
        """Update JIRA issue fields"""
        if not self._is_configured():
            logger.warning("JIRA service not configured")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {"fields": fields}
                
                response = await client.put(
                    f"{self.base_url}/rest/api/3/issue/{issue_key}",
                    json=payload,
                    auth=self.auth,
                    headers={"Accept": "application/json", "Content-Type": "application/json"}
                )
                
                if response.status_code == 204:
                    logger.info("JIRA issue updated successfully", issue_key=issue_key)
                    return True
                else:
                    logger.error("Failed to update JIRA issue", 
                               issue_key=issue_key,
                               status_code=response.status_code)
                    return False
                    
        except Exception as e:
            logger.error("Error updating JIRA issue", issue_key=issue_key, error=str(e))
            return False
    
    async def add_comment(self, issue_key: str, comment: str) -> bool:
        """Add comment to JIRA issue"""
        if not self._is_configured():
            logger.warning("JIRA service not configured")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "body": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "text": comment,
                                        "type": "text"
                                    }
                                ]
                            }
                        ]
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/rest/api/3/issue/{issue_key}/comment",
                    json=payload,
                    auth=self.auth,
                    headers={"Accept": "application/json", "Content-Type": "application/json"}
                )
                
                if response.status_code == 201:
                    logger.info("Comment added to JIRA issue", issue_key=issue_key)
                    return True
                else:
                    logger.error("Failed to add comment to JIRA issue", 
                               issue_key=issue_key,
                               status_code=response.status_code)
                    return False
                    
        except Exception as e:
            logger.error("Error adding comment to JIRA issue", issue_key=issue_key, error=str(e))
            return False
    
    async def get_acceptance_criteria(self, issue_key: str) -> Optional[str]:
        """Extract acceptance criteria from JIRA issue, cleaning markup and supporting custom fields."""
        issue_data = await self.get_issue(issue_key)
        if not issue_data:
            return None
        try:
            fields = issue_data.get("fields", {})
            description = fields.get("description")
            import re
            def extract_plain_text(desc):
                # Atlassian Document Format
                if isinstance(desc, dict) and "content" in desc:
                    text_parts = []
                    def extract(node):
                        if isinstance(node, dict):
                            if node.get("type") == "text":
                                text_parts.append(node.get("text", ""))
                            elif "content" in node:
                                for child in node["content"]:
                                    extract(child)
                        elif isinstance(node, list):
                            for item in node:
                                extract(item)
                    extract(desc["content"])
                    return " ".join(text_parts)
                elif isinstance(desc, str):
                    return desc
                return ""
            full_text = extract_plain_text(description)
            # Split on 'Acceptance Criteria' (case-insensitive)
            split = re.split(r'Acceptance Criteria[:\n]+', full_text, flags=re.IGNORECASE)
            ac = split[1].strip() if len(split) > 1 else ""
            # Remove Jira image/file markup
            ac = re.sub(r'!\S+?\.(jpg|png|jpeg|gif)[^!]*!', '', ac, flags=re.IGNORECASE).strip()
            # Remove Jira smart links and other link clutter: [text|url|smart-link] or [text|url]
            ac = re.sub(r'\[.*?\|.*?\]', '', ac)
            # Remove any trailing empty lines
            ac = "\n".join([line for line in ac.splitlines() if line.strip()])
            # Fallback to custom field if not found
            if not ac:
                custom_ac_field = fields.get("customfield_10000")
                if custom_ac_field:
                    return str(custom_ac_field).strip()
            return ac if ac else None
        except Exception as e:
            logger.error("Error extracting acceptance criteria", issue_key=issue_key, error=str(e))
            return None
    
    def _is_configured(self) -> bool:
        """Check if JIRA service is properly configured"""
        return bool(self.base_url and self.username and self.api_token)
