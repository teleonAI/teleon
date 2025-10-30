"""
GitHub Integration - Manage repositories, issues, PRs, and more.

Provides:
- Repository management
- Issue and PR operations
- Commit and branch operations
- Webhook handling
- Actions and workflows
"""

from typing import Dict, Any, List, Optional
import httpx

from teleon.integrations.base import (
    BaseIntegration,
    IntegrationConfig,
    IntegrationError,
    AuthenticationError,
)


class GitHubIntegration(BaseIntegration):
    """
    GitHub integration for repository automation.
    
    Example:
        >>> config = IntegrationConfig(
        ...     name="github",
        ...     api_key="ghp_your_token"
        ... )
        >>> github = GitHubIntegration(config)
        >>> issue = await github.create_issue("owner/repo", "Bug found", "Description")
    """
    
    def __init__(self, config: IntegrationConfig):
        """Initialize GitHub integration."""
        if not config.base_url:
            config.base_url = "https://api.github.com"
        super().__init__(config)
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            headers={
                "Authorization": f"token {config.api_key}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
    
    async def authenticate(self) -> bool:
        """Authenticate with GitHub API."""
        try:
            response = await self.client.get("/user")
            
            if response.status_code != 200:
                raise AuthenticationError(f"GitHub auth failed: {response.text}")
            
            self._authenticated = True
            self.logger.info("GitHub authentication successful")
            return True
        
        except Exception as e:
            raise AuthenticationError(f"GitHub authentication failed: {e}") from e
    
    async def test_connection(self) -> bool:
        """Test connection to GitHub."""
        return await self.authenticate()
    
    async def create_issue(
        self,
        repo: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a new issue.
        
        Args:
            repo: Repository in format "owner/repo"
            title: Issue title
            body: Issue description
            labels: List of labels
            assignees: List of assignees
            
        Returns:
            Created issue information
        """
        await self.ensure_authenticated()
        
        payload = {"title": title}
        
        if body:
            payload["body"] = body
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        
        async def _create():
            response = await self.client.post(
                f"/repos/{repo}/issues",
                json=payload
            )
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create issue: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def create_pull_request(
        self,
        repo: str,
        title: str,
        head: str,
        base: str,
        body: Optional[str] = None,
        draft: bool = False
    ) -> Dict[str, Any]:
        """
        Create a new pull request.
        
        Args:
            repo: Repository in format "owner/repo"
            title: PR title
            head: Name of the branch with your changes
            base: Name of the branch to merge into
            body: PR description
            draft: Whether this is a draft PR
            
        Returns:
            Created PR information
        """
        await self.ensure_authenticated()
        
        payload = {
            "title": title,
            "head": head,
            "base": base,
            "draft": draft
        }
        
        if body:
            payload["body"] = body
        
        async def _create():
            response = await self.client.post(
                f"/repos/{repo}/pulls",
                json=payload
            )
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create PR: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def list_repositories(
        self,
        org: Optional[str] = None,
        user: Optional[str] = None,
        visibility: str = "all",
        per_page: int = 30
    ) -> List[Dict[str, Any]]:
        """
        List repositories.
        
        Args:
            org: Organization name (if listing org repos)
            user: Username (if listing user repos)
            visibility: Repository visibility (all, public, private)
            per_page: Results per page
            
        Returns:
            List of repositories
        """
        await self.ensure_authenticated()
        
        if org:
            endpoint = f"/orgs/{org}/repos"
        elif user:
            endpoint = f"/users/{user}/repos"
        else:
            endpoint = "/user/repos"
        
        async def _list():
            response = await self.client.get(
                endpoint,
                params={"visibility": visibility, "per_page": per_page}
            )
            
            if response.status_code != 200:
                raise IntegrationError(f"Failed to list repos: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_list)
    
    async def get_commit(
        self,
        repo: str,
        commit_sha: str
    ) -> Dict[str, Any]:
        """
        Get commit information.
        
        Args:
            repo: Repository in format "owner/repo"
            commit_sha: Commit SHA
            
        Returns:
            Commit information
        """
        await self.ensure_authenticated()
        
        async def _get():
            response = await self.client.get(f"/repos/{repo}/commits/{commit_sha}")
            
            if response.status_code != 200:
                raise IntegrationError(f"Failed to get commit: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_get)
    
    async def create_webhook(
        self,
        repo: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a webhook for a repository.
        
        Args:
            repo: Repository in format "owner/repo"
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional webhook secret
            
        Returns:
            Created webhook information
        """
        await self.ensure_authenticated()
        
        payload = {
            "config": {
                "url": url,
                "content_type": "json"
            },
            "events": events,
            "active": True
        }
        
        if secret:
            payload["config"]["secret"] = secret
        
        async def _create():
            response = await self.client.post(
                f"/repos/{repo}/hooks",
                json=payload
            )
            
            if response.status_code not in [200, 201]:
                raise IntegrationError(f"Failed to create webhook: {response.text}")
            
            return response.json()
        
        return await self.execute_with_retry(_create)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

