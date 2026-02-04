"""Command-line interface for vault-embedder."""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import VaultConfig, load_config
from .indexer import VaultIndexer
from .search import VaultSearcher


console = Console()


@click.group()
@click.option('--config', '-c', 'config_path', help='Path to config file')
@click.pass_context
def cli(ctx, config_path: Optional[str]):
    """vault-embedder: Index and search your vault with local embeddings."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config_path


@cli.command()
@click.argument('files', nargs=-1)
@click.option('--force', '-f', is_flag=True, help='Force reindex even if unchanged')
@click.pass_context
def index(ctx, files: tuple, force: bool):
    """Index vault content.
    
    If FILES are specified, only index those files (relative paths).
    Otherwise, index the entire vault.
    """
    try:
        config = load_config(ctx.obj.get('config_path'))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    indexer = VaultIndexer(config)
    
    files_list = list(files) if files else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing...", total=None)
        
        def on_progress(file_path: str, current: int, total: int):
            progress.update(task, total=total, completed=current, description=f"Indexing {file_path}")
        
        result = indexer.index(files=files_list, force=force, progress_callback=on_progress)
    
    # Print results
    console.print()
    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Files processed", str(result.files_processed))
    table.add_row("Files skipped (unchanged)", str(result.files_skipped))
    table.add_row("Files deleted", str(result.files_deleted))
    table.add_row("Chunks added", str(result.chunks_added))
    table.add_row("Chunks removed", str(result.chunks_removed))
    
    console.print(table)
    
    if result.errors:
        console.print("\n[yellow]Errors:[/yellow]")
        for error in result.errors:
            console.print(f"  • {error}")


@cli.command()
@click.argument('query')
@click.option('--limit', '-n', default=10, help='Maximum number of results')
@click.option('--min-score', '-s', default=0.0, help='Minimum similarity score (0-1)')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.option('--path', '-p', 'path_filter', help='Filter by path prefix')
@click.pass_context
def search(ctx, query: str, limit: int, min_score: float, output_json: bool, path_filter: Optional[str]):
    """Search indexed vault content."""
    try:
        config = load_config(ctx.obj.get('config_path'))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    searcher = VaultSearcher(config)
    
    with console.status("Searching..."):
        results = searcher.search(
            query=query,
            limit=limit,
            min_score=min_score,
            path_filter=path_filter,
        )
    
    if output_json:
        output = {
            "query": query,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
        return
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    console.print(f"\n[bold]Results for:[/bold] {query}\n")
    
    for i, result in enumerate(results, 1):
        score_color = "green" if result.score > 0.7 else "yellow" if result.score > 0.5 else "red"
        
        console.print(f"[bold cyan]{i}.[/bold cyan] [{score_color}]{result.score:.3f}[/{score_color}] [bold]{result.path}[/bold]")
        if result.heading:
            console.print(f"   [dim]Section: {result.heading}[/dim]")
        console.print(f"   [dim]Lines {result.line_start}-{result.line_end}[/dim]")
        
        # Truncate text for display
        text = result.text[:300] + "..." if len(result.text) > 300 else result.text
        console.print(f"   {text}")
        console.print()


@cli.command()
@click.pass_context
def status(ctx):
    """Show index status."""
    try:
        config = load_config(ctx.obj.get('config_path'))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    indexer = VaultIndexer(config)
    info = indexer.status()
    
    table = Table(title="Vault Index Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Collection", info['collection_name'])
    table.add_row("Vault Path", info['vault_path'])
    table.add_row("Model", info['model_name'])
    table.add_row("Indexed Files", str(info['indexed_files']))
    table.add_row("Total Chunks", str(info['total_chunks']))
    
    if info['last_indexed']:
        from datetime import datetime
        last_indexed = datetime.fromtimestamp(info['last_indexed'])
        table.add_row("Last Indexed", last_indexed.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        table.add_row("Last Indexed", "Never")
    
    console.print(table)


@cli.command()
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation')
@click.pass_context
def clear(ctx, yes: bool):
    """Clear all indexed data."""
    try:
        config = load_config(ctx.obj.get('config_path'))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    if not yes:
        if not click.confirm(f"This will delete all indexed data for collection '{config.collection_name}'. Continue?"):
            console.print("[yellow]Aborted.[/yellow]")
            return
    
    indexer = VaultIndexer(config)
    indexer.clear()
    
    console.print(f"[green]Cleared collection '{config.collection_name}'[/green]")


@cli.command()
@click.argument('file_path')
@click.pass_context
def delete(ctx, file_path: str):
    """Delete a specific file from the index."""
    try:
        config = load_config(ctx.obj.get('config_path'))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    indexer = VaultIndexer(config)
    count = indexer.delete_file(file_path)
    
    if count:
        console.print(f"[green]Deleted {count} chunks for '{file_path}'[/green]")
    else:
        console.print(f"[yellow]No indexed content found for '{file_path}'[/yellow]")


@cli.command()
@click.argument('source_path')
@click.option('--limit', '-n', default=10, help='Maximum number of results')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
@click.pass_context
def similar(ctx, source_path: str, limit: int, output_json: bool):
    """Find content similar to a specific file."""
    try:
        config = load_config(ctx.obj.get('config_path'))
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    
    searcher = VaultSearcher(config)
    
    with console.status("Searching..."):
        results = searcher.search_by_path(
            source_path=source_path,
            limit=limit,
        )
    
    if output_json:
        output = {
            "source": source_path,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2))
        return
    
    if not results:
        console.print("[yellow]No similar content found.[/yellow]")
        return
    
    console.print(f"\n[bold]Similar to:[/bold] {source_path}\n")
    
    for i, result in enumerate(results, 1):
        score_color = "green" if result.score > 0.7 else "yellow" if result.score > 0.5 else "red"
        
        console.print(f"[bold cyan]{i}.[/bold cyan] [{score_color}]{result.score:.3f}[/{score_color}] [bold]{result.path}[/bold]")
        if result.heading:
            console.print(f"   [dim]Section: {result.heading}[/dim]")
        
        text = result.text[:200] + "..." if len(result.text) > 200 else result.text
        console.print(f"   {text}")
        console.print()


def main():
    """Entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
