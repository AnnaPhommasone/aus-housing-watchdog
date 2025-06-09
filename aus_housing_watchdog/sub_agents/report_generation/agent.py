"""
Report Agent for Aus Housing Watchdog

Responsible for generating reports and visualizations from housing market analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import json
from google.cloud import storage
from google.adk.agents import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportAgent:
    """Agent responsible for generating reports and visualizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Report Agent.
        
        Args:
            config: Configuration dictionary with report settings
        """
        self.agent = Agent(
            name="report_agent",
            model="gemini-2.0-flash",
            description="Generates reports and visualizations from housing market analysis",
            instruction="""
            You are an expert data visualization and reporting specialist. Your role is to 
            create clear, insightful, and visually appealing reports from housing market data,
            including charts, tables, and narrative summaries that highlight key findings.
            """,
            tools=[
                self.generate_market_report,
                self.plot_time_series,
                self.plot_geographic,
                self.plot_distribution,
                self.export_report
            ]
        )
        
        # Initialize configuration
        self.config = config or {}
        self.gcs_client = storage.Client() if self.config.get('use_gcs', False) else None
        
        # Default report settings
        self.default_settings = {
            'theme': 'plotly_white',
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent': '#2ca02c',
                'highlight': '#d62728',
                'background': '#f8f9fa',
                'text': '#212529'
            },
            'font_family': 'Arial, sans-serif',
            'templates': {
                'market_report': 'templates/market_report.html',
                'email_summary': 'templates/email_summary.html'
            },
            'export': {
                'formats': ['html', 'pdf', 'png'],
                'gcs_bucket': 'aus-housing-reports',
                'local_path': 'reports'
            }
        }
        
        # Merge with user-provided settings
        self.settings = {**self.default_settings, **self.config.get('report_settings', {})}
        
        # Ensure local directories exist
        if 'local_path' in self.settings['export']:
            Path(self.settings['export']['local_path']).mkdir(parents=True, exist_ok=True)
    
    def generate_market_report(self, 
                             data: Dict[str, Any], 
                             report_type: str = 'market_summary',
                             output_format: str = 'html') -> Dict[str, Any]:
        """Generate a comprehensive market report.
        
        Args:
            data: Dictionary containing analysis results
            report_type: Type of report to generate
            output_format: Output format ('html', 'pdf', 'markdown')
            
        Returns:
            Dictionary with report content and metadata
        """
        try:
            # Prepare report content based on report type
            if report_type == 'market_summary':
                content = self._generate_market_summary(data, output_format)
            elif report_type == 'anomaly_report':
                content = self._generate_anomaly_report(data, output_format)
            elif report_type == 'trend_analysis':
                content = self._generate_trend_report(data, output_format)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
            
            # Generate report metadata
            report_id = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            timestamp = datetime.utcnow().isoformat()
            
            # Prepare response
            report = {
                'report_id': report_id,
                'type': report_type,
                'format': output_format,
                'generated_at': timestamp,
                'content': content
            }
            
            # Export if requested
            if output_format != 'dict':
                export_path = self.export_report(
                    content, 
                    report_id, 
                    output_format,
                    data.get('export_options', {})
                )
                report['export_path'] = export_path
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating {report_type} report: {e}")
            raise
    
    def _generate_market_summary(self, data: Dict[str, Any], output_format: str) -> Any:
        """Generate a market summary report."""
        # Extract key metrics from data
        summary = data.get('summary', {})
        current = summary.get('current_period', {})
        
        # Create a simple text summary
        text_content = f"""
        # Housing Market Summary Report
        
        ## Overview
        - **Period**: {start_date} to {end_date}
        - **Total Listings**: {n_listings:,}
        - **Average Price**: ${avg_price:,.0f}
        - **Median Price**: ${median_price:,.0f}
        - **Price Range**: ${min_price:,.0f} - ${max_price:,.0f}
        
        ## Market Trends
        - **Year-over-Year Change**: {yoy_change:+.1f}%
        - **Market Trend**: {trend_direction} ({trend_strength:.1f} strength)
        
        ## Key Insights
        {insights}
        """.format(
            start_date=current.get('start', 'N/A'),
            end_date=current.get('end', 'N/A'),
            n_listings=current.get('n_listings', 0),
            avg_price=current.get('avg_value', 0),
            median_price=current.get('median_value', 0),
            min_price=current.get('min_value', 0),
            max_price=current.get('max_value', 0),
            yoy_change=summary.get('yoy_change_percent', 0),
            trend_direction=summary.get('trend_analysis', {}).get('trend_direction', 'stable').capitalize(),
            trend_strength=summary.get('trend_analysis', {}).get('trend_strength', 0) * 100,
            insights=self._generate_insights(data)
        )
        
        # Generate visualizations
        visualizations = {}
        
        # Price trend chart
        if 'time_series' in data:
            fig = self.plot_time_series(
                data['time_series'],
                x='date',
                y='median_price',
                title='Median Housing Prices Over Time',
                return_figure=True
            )
            visualizations['price_trend'] = fig
        
        # Price distribution
        if 'price_distribution' in data:
            fig = self.plot_distribution(
                data['price_distribution'],
                x='price',
                title='Price Distribution',
                return_figure=True
            )
            visualizations['price_distribution'] = fig
        
        # Format based on output format
        if output_format == 'html':
            return self._format_html_report(text_content, visualizations)
        elif output_format == 'markdown':
            return self._format_markdown_report(text_content, visualizations)
        else:  # dict
            return {
                'text': text_content,
                'visualizations': visualizations,
                'metrics': {
                    'avg_price': current.get('avg_value'),
                    'median_price': current.get('median_value'),
                    'yoy_change': summary.get('yoy_change_percent'),
                    'trend': summary.get('trend_analysis', {})
                }
            }
    
    def _generate_anomaly_report(self, data: Dict[str, Any], output_format: str) -> Any:
        """Generate an anomaly detection report."""
        # Implementation for anomaly report
        pass
    
    def _generate_trend_report(self, data: Dict[str, Any], output_format: str) -> Any:
        """Generate a trend analysis report."""
        # Implementation for trend report
        pass
    
    def _generate_insights(self, data: Dict[str, Any]) -> str:
        """Generate narrative insights from analysis results."""
        insights = []
        
        # Market trend insight
        trend = data.get('summary', {}).get('trend_analysis', {})
        if trend.get('trend_direction') == 'increasing' and trend.get('trend_strength', 0) > 0.5:
            insights.append("The market is showing a strong upward trend in prices.")
        elif trend.get('trend_direction') == 'decreasing' and trend.get('trend_strength', 0) > 0.5:
            insights.append("The market is experiencing a notable decline in prices.")
        else:
            insights.append("The market is relatively stable with no strong directional trend.")
        
        # Price distribution insight
        if 'price_distribution' in data:
            prices = data['price_distribution'].get('prices', [])
            if prices:
                price_range = max(prices) - min(prices)
                if price_range > 1000000:  # Arbitrary threshold
                    insights.append("There is a wide range of property prices in the market.")
        
        # Add more insights based on available data
        if 'anomalies' in data and data['anomalies']:
            insights.append(f"Detected {len(data['anomalies'])} potential anomalies in the data that may require investigation.")
        
        return "\n\n".join(f"- {insight}" for insight in insights) if insights else "No significant insights available."
    
    def plot_time_series(self, 
                       data: Union[pd.DataFrame, Dict], 
                       x: str, 
                       y: str,
                       title: str = None,
                       return_figure: bool = False):
        """Create a time series plot.
        
        Args:
            data: Input data (DataFrame or dict)
            x: Column name for x-axis (typically date)
            y: Column name for y-axis
            title: Plot title
            return_figure: If True, return the Plotly figure object
            
        Returns:
            Plotly figure or HTML div
        """
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            fig = px.line(
                df, 
                x=x, 
                y=y,
                title=title or f"{y} Over Time",
                template=self.settings['theme']
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title=x.capitalize(),
                yaxis_title=y.replace('_', ' ').title(),
                hovermode='x unified',
                font_family=self.settings['font_family'],
                plot_bgcolor=self.settings['colors']['background']
            )
            
            if return_figure:
                return fig
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return f"<p>Error generating plot: {str(e)}</p>"
    
    def plot_geographic(self, 
                      data: Union[pd.DataFrame, Dict], 
                      lat: str = 'latitude',
                      lon: str = 'longitude',
                      color: str = None,
                      size: str = None,
                      title: str = None,
                      return_figure: bool = False):
        """Create a geographic plot.
        
        Args:
            data: Input data with geographic coordinates
            lat: Column name for latitude
            lon: Column name for longitude
            color: Column to use for color encoding
            size: Column to use for size encoding
            title: Plot title
            return_figure: If True, return the Plotly figure object
            
        Returns:
            Plotly figure or HTML div
        """
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            fig = px.scatter_mapbox(
                df,
                lat=lat,
                lon=lon,
                color=color,
                size=size,
                title=title or 'Geographic Distribution',
                hover_data=df.columns.tolist(),
                zoom=10,
                height=600
            )
            
            # Use OpenStreetMap tiles
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0, "t":30, "l":0, "b":0},
                title_x=0.5,
                font_family=self.settings['font_family']
            )
            
            if return_figure:
                return fig
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating geographic plot: {e}")
            return f"<p>Error generating map: {str(e)}</p>"
    
    def plot_distribution(self, 
                        data: Union[pd.DataFrame, Dict],
                        x: str,
                        title: str = None,
                        color: str = None,
                        bins: int = None,
                        return_figure: bool = False):
        """Create a distribution plot.
        
        Args:
            data: Input data
            x: Column to plot distribution for
            title: Plot title
            color: Column to use for color encoding
            bins: Number of bins for histogram
            return_figure: If True, return the Plotly figure object
            
        Returns:
            Plotly figure or HTML div
        """
        try:
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            fig = px.histogram(
                df, 
                x=x, 
                color=color,
                nbins=bins,
                title=title or f"Distribution of {x.replace('_', ' ').title()}",
                template=self.settings['theme']
            )
            
            # Customize layout
            fig.update_layout(
                xaxis_title=x.replace('_', ' ').title(),
                yaxis_title='Count',
                bargap=0.1,
                font_family=self.settings['font_family'],
                plot_bgcolor=self.settings['colors']['background']
            )
            
            if return_figure:
                return fig
            return fig.to_html(full_html=False, include_plotlyjs='cdn')
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {e}")
            return f"<p>Error generating distribution plot: {str(e)}</p>"
    
    def export_report(self, 
                     content: Any, 
                     report_id: str, 
                     output_format: str = 'html',
                     options: Optional[Dict] = None) -> Dict[str, str]:
        """Export a report to the specified format.
        
        Args:
            content: Report content
            report_id: Unique identifier for the report
            output_format: Export format ('html', 'pdf', 'png')
            options: Additional export options
            
        Returns:
            Dictionary with export paths and metadata
        """
        options = options or {}
        export_paths = {}
        
        try:
            # HTML export
            if output_format == 'html':
                if not isinstance(content, str):
                    content = str(content)
                
                # Save locally
                local_path = Path(self.settings['export']['local_path']) / f"{report_id}.html"
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                export_paths['local'] = str(local_path)
                
                # Upload to GCS if configured
                if self.gcs_client:
                    gcs_bucket = options.get('gcs_bucket') or self.settings['export'].get('gcs_bucket')
                    if gcs_bucket:
                        blob_name = f"reports/{report_id}.html"
                        bucket = self.gcs_client.bucket(gcs_bucket)
                        blob = bucket.blob(blob_name)
                        blob.upload_from_string(content, content_type='text/html')
                        export_paths['gcs'] = f"gs://{gcs_bucket}/{blob_name}"
            
            # PDF/PNG export (for visualizations)
            elif output_format in ['pdf', 'png']:
                # This would require additional dependencies like kaleido
                # For now, just save the HTML with a note
                local_path = Path(self.settings['export']['local_path']) / f"{report_id}.html"
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(f"<p>Export to {output_format.upper()} is not yet implemented. Saving as HTML instead.</p>\n")
                    f.write(content)
                export_paths['local'] = str(local_path)
                export_paths['note'] = f"{output_format.upper()} export not implemented"
            
            else:
                raise ValueError(f"Unsupported export format: {output_format}")
            
            return {
                'status': 'success',
                'report_id': report_id,
                'format': output_format,
                'paths': export_paths,
                'exported_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'report_id': report_id,
                'format': output_format
            }
    
    def _format_html_report(self, text_content: str, visualizations: Dict[str, Any]) -> str:
        """Format a report as HTML."""
        html_parts = [
            """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Housing Market Report</title>
                <style>
                    body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin-bottom: 30px; }
                    .visualization { margin: 20px 0; border: 1px solid #eee; padding: 15px; border-radius: 5px; }
                    h1 { color: #1f77b4; }
                    h2 { color: #2ca02c; border-bottom: 1px solid #eee; padding-bottom: 5px; }
                    .insights { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #ff7f0e; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Housing Market Analysis Report</h1>
                        <p>Generated on {date}</p>
                    </div>
            """.format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            text_content.replace('\n', '<br>').replace('## ', '<h2>').replace('\n', '</h2>\n'),
        ]
        
        # Add visualizations
        if visualizations:
            html_parts.append('<div class="section"><h2>Visualizations</h2>')
            for name, viz in visualizations.items():
                if hasattr(viz, 'to_html'):
                    html_parts.append(f'<div class="visualization"><h3>{name.replace("_", " ").title()}</h3>')
                    html_parts.append(viz.to_html(full_html=False, include_plotlyjs='cdn'))
                    html_parts.append('</div>')
            html_parts.append('</div>')
        
        # Close HTML
        html_parts.extend([
            '</div>',
            '    </body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)
    
    def _format_markdown_report(self, text_content: str, visualizations: Dict[str, Any]) -> str:
        """Format a report as Markdown."""
        # For markdown, we can include image references or data tables
        # Visualizations would need to be saved as images first
        markdown = [text_content]
        
        if visualizations:
            markdown.append('\n## Visualizations\n')
            for name, viz in visualizations.items():
                markdown.append(f'### {name.replace("_", " ").title()}\n')
                markdown.append(f'*[Figure: {name} - export as image for details]*\n')
        
        return '\n'.join(markdown)
