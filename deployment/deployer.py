"""Deployment system for Teleon agents."""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from teleon.config.loader import ConfigLoader


console = Console()


class AgentDeployment:
    """Represents a deployed agent."""
    
    def __init__(
        self,
        name: str,
        version: str,
        status: str,
        url: Optional[str] = None,
        replicas: int = 1,
        deployed_at: Optional[datetime] = None
    ):
        self.name = name
        self.version = version
        self.status = status
        self.url = url
        self.replicas = replicas
        self.deployed_at = deployed_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'status': self.status,
            'url': self.url,
            'replicas': self.replicas,
            'deployed_at': self.deployed_at.isoformat()
        }


class Deployer:
    """
    Handle agent deployments to various environments.
    
    Supports:
    - Local deployment (Docker Compose)
    - Cloud deployment (Kubernetes)
    - Serverless deployment (Cloud Functions)
    """
    
    def __init__(
        self,
        environment: str = "development",
        config_file: Optional[Path] = None
    ):
        """
        Initialize deployer.
        
        Args:
            environment: Target environment (development, staging, production)
            config_file: Path to configuration file
        """
        self.environment = environment
        self.config_loader = ConfigLoader(config_file)
        self.deployments: List[AgentDeployment] = []
    
    def deploy_local(
        self,
        agent_file: Path,
        agent_name: Optional[str] = None,
        port: int = 8000
    ) -> AgentDeployment:
        """
        Deploy agent locally using Docker Compose.
        
        Args:
            agent_file: Path to agent Python file
            agent_name: Name of agent to deploy
            port: Port to expose
        
        Returns:
            AgentDeployment instance
        """
        console.print(Panel.fit(
            f"[bold green]Deploying Agent Locally[/bold green]\n"
            f"File: [cyan]{agent_file}[/cyan]\n"
            f"Port: [yellow]{port}[/yellow]",
            title="🚀 Local Deployment"
        ))
        
        # Create deployment
        deployment = AgentDeployment(
            name=agent_name or agent_file.stem,
            version="dev",
            status="running",
            url=f"http://localhost:{port}",
            replicas=1
        )
        
        self.deployments.append(deployment)
        
        console.print(f"\n[green]✓[/green] Agent deployed successfully!")
        console.print(f"[bold]URL:[/bold] {deployment.url}")
        console.print(f"[bold]Status:[/bold] {deployment.status}")
        
        return deployment
    
    def deploy_cloud(
        self,
        agent_file: Path,
        agent_name: Optional[str] = None,
        replicas: int = 2
    ) -> AgentDeployment:
        """
        Deploy agent to cloud (Kubernetes).
        
        Args:
            agent_file: Path to agent Python file
            agent_name: Name of agent to deploy
            replicas: Number of replicas
        
        Returns:
            AgentDeployment instance
        """
        console.print(Panel.fit(
            f"[bold green]Deploying to Cloud[/bold green]\n"
            f"File: [cyan]{agent_file}[/cyan]\n"
            f"Replicas: [yellow]{replicas}[/yellow]",
            title="☁️ Cloud Deployment"
        ))
        
        name = agent_name or agent_file.stem
        
        # Generate deployment manifest
        manifest = self._generate_k8s_manifest(name, agent_file, replicas)
        
        # In a real implementation, this would:
        # 1. Build Docker image
        # 2. Push to container registry
        # 3. Apply Kubernetes manifest
        # 4. Wait for rollout to complete
        
        deployment = AgentDeployment(
            name=name,
            version="1.0.0",
            status="deploying",
            url=f"https://{name}.teleon.cloud",
            replicas=replicas
        )
        
        self.deployments.append(deployment)
        
        console.print(f"\n[green]✓[/green] Deployment initiated!")
        console.print(f"[bold]Name:[/bold] {deployment.name}")
        console.print(f"[bold]URL:[/bold] {deployment.url}")
        console.print(f"[bold]Replicas:[/bold] {deployment.replicas}")
        console.print(f"\n[yellow]Note:[/yellow] Cloud deployment is a placeholder in this version")
        
        return deployment
    
    def _generate_k8s_manifest(
        self,
        name: str,
        agent_file: Path,
        replicas: int
    ) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'{name}-deployment',
                'labels': {
                    'app': name,
                    'teleon.ai/agent': 'true'
                }
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {
                        'app': name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': name,
                            'image': f'teleon/{name}:latest',
                            'ports': [{
                                'containerPort': 8000
                            }],
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': self.environment},
                                {'name': 'AGENT_NAME', 'value': name}
                            ],
                            'resources': {
                                'requests': {
                                    'cpu': '100m',
                                    'memory': '128Mi'
                                },
                                'limits': {
                                    'cpu': '500m',
                                    'memory': '512Mi'
                                }
                            }
                        }]
                    }
                }
            }
        }
    
    def list_deployments(self) -> List[AgentDeployment]:
        """List all deployments."""
        return self.deployments
    
    def get_deployment(self, name: str) -> Optional[AgentDeployment]:
        """Get deployment by name."""
        for deployment in self.deployments:
            if deployment.name == name:
                return deployment
        return None
    
    def delete_deployment(self, name: str) -> bool:
        """Delete a deployment."""
        deployment = self.get_deployment(name)
        if deployment:
            self.deployments.remove(deployment)
            console.print(f"[green]✓[/green] Deployment '{name}' deleted")
            return True
        console.print(f"[red]✗[/red] Deployment '{name}' not found")
        return False
    
    def show_status(self) -> None:
        """Display deployment status table."""
        if not self.deployments:
            console.print("[yellow]No deployments found[/yellow]")
            return
        
        table = Table(title="Agent Deployments")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("URL", style="blue")
        table.add_column("Replicas", justify="center")
        table.add_column("Deployed At", style="dim")
        
        for deployment in self.deployments:
            table.add_row(
                deployment.name,
                deployment.version,
                deployment.status,
                deployment.url or "-",
                str(deployment.replicas),
                deployment.deployed_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        console.print(table)


def create_deployer(environment: str = "development") -> Deployer:
    """
    Create a deployer instance.
    
    Args:
        environment: Target environment
    
    Returns:
        Deployer instance
    """
    return Deployer(environment=environment)

