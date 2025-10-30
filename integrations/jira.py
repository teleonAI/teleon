"""
Jira Integration - Manage tickets, projects, and workflows.

Provides:
- Create and update issues
- Project management
- Sprint operations
- Search and JQL queries
- Workflow automation
"""

from typing import Dict, Any, List, Optional
import httpx

from teleon.integrations.base import (
    BaseIntegration,
    IntegrationConfig,
    IntegrationError,
    AuthenticationError,
)


class JiraIntegration(BaseIntegration):
    """
    Jira integration for issue tracking and project management.
    
    Example:
        >>> config = IntegrationConfig(
        ...     name="jira",
        ...     base_url="https://your-domain.atlassian.net",
        ...     api_key="your-api-token",
        ...     extra={"email": "your-email@example.com"}
        ... )
        >>> jira = JiraIntegration(config)
        >>> issue = await jira.create_issue("PROJ", "Bug", "Title", "Description")
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize Jira integration."""
        if not config.base_url:
            raise IntegrationError("Jira requires base_url (e.g., https://your-domain.atlassian.net)")
        
        super().__init__(config)
        
        # Jira uses email + API token for auth
        email = config.extra.get("email")
        if not email:
            raise IntegrationError("Jira requires 'email' in extra config")
        
        self.client = httpx.AsyncClient(
            base_url=f"{config.base_url}/rest/api/3",
            timeout=config.timeout,
            auth=(email, config.api_key)
        )
    
    async def authenticate(self) -> bool:
        """Authenticate with Jira API."""
        try:
            response = await self.client.get("/myself")
            
            if response.status_code != 200:
                raise AuthenticationError(f"Jira auth failed: {response.text}")
            
            self._authenticated = True
            self.logger.info("Jira authentication successful")
            return True
        
        except Exception as e:
            raise AuthenticationError(f"Jira authentication failed: {e}") from e
    
    async def test_connection(self) -> bool:
        """Test connection to Jira."""
        return await self.authenticate()
    
    async def create_issue(
        self,
        project_key: str,
        issue_type: str,
        summary: str,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new Jira issue.
        
        Args:
            project_key: Project key (e.g., "PROJ")
            issue_type: Issue type (e.g., "Bug", "Task", "Story")
            summary: Issue summary
            description: Issue description
            priority: Priority (e.g., "High", "Medium", "Low")
            labels: List of labels
            assignee: Assignee account ID
            
        Returns:
            Created issue information
        """
        await self.ensure_authenticated()
        
        fields = {
            "project": {"key": project_key},
            "issuetype": {"name": issue_type},
            "summary": summary
        }
        
        if description:
            fields["description"] = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }
                ]
            }
        
        if priority:
            fields["priority"] = {"name": priority}
        
        if labels:
            fields["labels"] = labels
        
        if assignee:
            fields["assignee"] = {"accountId": assignee}
        
        payload = {"fields": fields}
        
        async def _create():
            response = await self.client.post("/issue", json=payload)
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create issue: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def update_issue(
        self,
        issue_key: str,
        fields: Dict[str, Any]
    ) -> bool:
        """
        Update an existing issue.
        
        Args:
            issue_key: Issue key (e.g., "PROJ-123")
            fields: Fields to update
            
        Returns:
            True if successful
        """
        await self.ensure_authenticated()
        
        payload = {"fields": fields}
        
        async def _update():
            response = await self.client.put(f"/issue/{issue_key}", json=payload)
            
            if response.status_code != 204:
                raise IntegrationError(f"Failed to update issue: {response.text}")
            
            return True
        
        return await self.execute_with_retry(_update)
    
    async def add_comment(
        self,
        issue_key: str,
        comment: str
    ) -> Dict[str, Any]:
        """
        Add a comment to an issue.
        
        Args:
            issue_key: Issue key
            comment: Comment text
            
        Returns:
            Created comment information
        """
        await self.ensure_authenticated()
        
        payload = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": comment}]
                    }
                ]
            }
        }
        
        async def _add():
            response = await self.client.post(
                f"/issue/{issue_key}/comment",
                json=payload
            )
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to add comment: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_add)
    
    async def search_issues(
        self,
        jql: str,
        max_results: int = 50,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for issues using JQL.
        
        Args:
            jql: JQL query string
            max_results: Maximum number of results
            fields: Fields to return
            
        Returns:
            List of matching issues
        """
        await self.ensure_authenticated()
        
        params = {
            "jql": jql,
            "maxResults": max_results
        }
        
        if fields:
            params["fields"] = ",".join(fields)
        
        async def _search():
            response = await self.client.get("/search", params=params)
            
            if response.status_code != 200:
                raise IntegrationError(f"Failed to search issues: {response.text}")
            
            data = response.json()
            return data.get("issues", [])
        
        return await self.execute_with_retry(_search)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

