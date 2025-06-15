"""
Tools for the NSW Housing Recommendation Agent.

This file contains functions for generating personalized housing recommendations
based on analyzed NSW housing market data and user profiles.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from vertexai.generative_models import CallbackContext

# Helper function to load data and analysis results if needed
def get_analysis_results(callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Retrieve the analysis results from the state.
    
    Args:
        callback_context: The callback context containing state information.
    
    Returns:
        Dictionary containing analysis results.
    """
    analysis_results = callback_context.state.get("analysis_results", {})
    if not analysis_results:
        print("Warning: No analysis results found in state.")
    return analysis_results

def generate_suburb_recommendations(user_profile: Dict[str, Any], callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Generate personalized suburb recommendations based on user profile and analysis results.
    
    Args:
        user_profile: Dictionary containing user profile information
        callback_context: The callback context containing state information
    
    Returns:
        Dictionary containing recommended suburbs with justifications
    """
    recommendations = {}
    
    try:
        # Get analysis results from state
        analysis_results = get_analysis_results(callback_context)
        
        # Check if we have user matching results
        if not analysis_results or 'user_matches' not in analysis_results:
            return {"error": "Missing user matching analysis results"}
        
        user_matches = analysis_results.get('user_matches', {})
        price_trends = analysis_results.get('price_trends', {})
        location_analysis = analysis_results.get('location_analysis', {})
        property_type_analysis = analysis_results.get('property_type_analysis', {})
        
        # Get recommended localities from user matching (if available)
        recommended_localities = user_matches.get('recommended_localities', [])
        
        # If we don't have recommendations from user matching, get top affordable suburbs
        if not recommended_localities and 'location_analysis' in analysis_results:
            affordable_localities = location_analysis.get('most_affordable_localities', {})
            if affordable_localities:
                recommended_localities = list(affordable_localities.keys())[:5]
        
        # Create justification for each recommended locality
        suburb_recommendations = []
        for i, locality in enumerate(recommended_localities[:10]):  # Limit to top 10
            justifications = []
            
            # Check if this locality is within budget
            if user_profile.get('budget') and 'most_affordable_localities' in location_analysis:
                if locality in location_analysis.get('most_affordable_localities', {}):
                    median_price = location_analysis['most_affordable_localities'].get(locality)
                    if median_price and median_price <= user_profile.get('budget', 0) * 1.1:
                        justifications.append(f"Median price (${median_price:,.2f}) is within your budget")
            
            # Check for growth potential
            if 'growth_by_locality' in location_analysis and 'fastest_growing' in location_analysis.get('growth_by_locality', {}):
                if locality in location_analysis['growth_by_locality'].get('fastest_growing', {}):
                    growth_rate = location_analysis['growth_by_locality']['fastest_growing'].get(locality)
                    justifications.append(f"Strong growth potential ({growth_rate:.1f}%)")
            
            # Add recommended property types for this locality
            recommended_property_types = []
            if 'property_type_by_locality' in property_type_analysis:
                property_types = property_type_analysis.get('property_type_by_locality', {}).get(locality, {})
                if property_types:
                    recommended_property_types = list(property_types.keys())[:3]  # Top 3 property types
            
            # Add any anomalies or warnings (if applicable)
            warnings = []
            for anomaly in analysis_results.get('market_anomalies', []):
                if 'affected_localities' in anomaly and locality in anomaly.get('affected_localities', []):
                    warnings.append(anomaly.get('description', ''))
            
            # Create the recommendation object
            recommendation = {
                'suburb': locality,
                'rank': i + 1,
                'justifications': justifications,
                'recommended_property_types': recommended_property_types,
                'warnings': warnings
            }
            
            suburb_recommendations.append(recommendation)
        
        recommendations['suburb_recommendations'] = suburb_recommendations
        
        # Store the results in state
        callback_context.state["recommendations"]["suburbs"] = suburb_recommendations
        
        return recommendations
    except Exception as e:
        print(f"Error generating suburb recommendations: {str(e)}")
        return {"error": str(e)}

def generate_property_recommendations(user_profile: Dict[str, Any], callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Generate personalized property type recommendations based on user profile and analysis results.
    
    Args:
        user_profile: Dictionary containing user profile information
        callback_context: The callback context containing state information
    
    Returns:
        Dictionary containing recommended property types with justifications
    """
    recommendations = {}
    
    try:
        # Get analysis results from state
        analysis_results = get_analysis_results(callback_context)
        
        # Check if we have property type analysis
        if not analysis_results or 'property_type_analysis' not in analysis_results:
            return {"error": "Missing property type analysis results"}
        
        property_type_analysis = analysis_results.get('property_type_analysis', {})
        price_trends = analysis_results.get('price_trends', {})
        
        # Get user preferences
        family_size = user_profile.get('family_size', 1)
        budget = user_profile.get('budget', 0)
        investment_goal = user_profile.get('investment_goal', 'both')
        
        # Get property type distribution and price stats
        property_distribution = property_type_analysis.get('property_type_distribution', {})
        property_price_stats = property_type_analysis.get('property_price_by_type', {})
        property_trends = price_trends.get('price_by_property_type', {})
        
        # Determine suitable property types based on user profile
        suitable_property_types = []
        
        # Logic for family size
        if family_size >= 4:
            # Large families need houses
            primary_types = ['house', 'villa']
            secondary_types = ['townhouse']
        elif family_size >= 2:
            # Medium families can use houses or townhouses
            primary_types = ['house', 'townhouse']
            secondary_types = ['villa', 'unit']
        else:
            # Singles or couples might prefer apartments or smaller properties
            primary_types = ['unit', 'apartment']
            secondary_types = ['townhouse', 'studio']
        
        # Check for available property types in our data
        available_types = list(property_distribution.keys())
        
        # Filter property types based on budget
        affordable_types = []
        if budget > 0:
            for prop_type, stats in property_price_stats.items():
                if prop_type in available_types:
                    median_price = stats.get('median')
                    if median_price and median_price <= budget * 1.1:  # 10% buffer
                        affordable_types.append(prop_type)
        
        # Generate recommendations with justifications
        property_recommendations = []
        
        # Combine primary and secondary types prioritizing primary
        candidate_types = []
        for pt in primary_types:
            if pt in available_types:
                candidate_types.append((pt, 'primary'))
                
        for pt in secondary_types:
            if pt in available_types and pt not in [c[0] for c in candidate_types]:
                candidate_types.append((pt, 'secondary'))
                
        # Add any affordable types not already included
        for pt in affordable_types:
            if pt not in [c[0] for c in candidate_types]:
                candidate_types.append((pt, 'budget'))
        
        # Generate recommendations for each candidate type
        for prop_type, category in candidate_types[:5]:  # Top 5 recommendations
            justifications = []
            
            # Budget justification
            if prop_type in affordable_types:
                median_price = property_price_stats.get(prop_type, {}).get('median', 0)
                justifications.append(f"Median price (${median_price:,.2f}) is within your budget")
            
            # Family size justification
            if category == 'primary':
                if family_size >= 4:
                    justifications.append("Suitable for your large family size")
                elif family_size >= 2:
                    justifications.append("Well-suited for your family size")
                else:
                    justifications.append("Ideal for singles or couples")
            elif category == 'secondary':
                justifications.append("Could accommodate your needs with some compromises")
            
            # Investment goal justification
            if investment_goal and prop_type in property_trends:
                growth_rate = property_trends.get(prop_type, {}).get('growth_rate', 0)
                if investment_goal == 'capital_growth' and growth_rate > 5:
                    justifications.append(f"Strong capital growth potential ({growth_rate:.1f}%)")
                elif investment_goal == 'rental_yield':
                    # Add rental yield info if available
                    if 'rental_yield' in property_type_analysis:
                        yield_data = property_type_analysis.get('rental_yield', {}).get(prop_type)
                        if yield_data:
                            justifications.append(f"Good rental yield potential ({yield_data:.1f}%)")
            
            # Add the recommendation
            recommendation = {
                'property_type': prop_type,
                'category': category,
                'justifications': justifications,
                'median_price': property_price_stats.get(prop_type, {}).get('median', 0),
                'distribution_percentage': property_distribution.get(prop_type, 0)
            }
            
            property_recommendations.append(recommendation)
        
        recommendations['property_type_recommendations'] = property_recommendations
        
        # Store in state
        callback_context.state["recommendations"]["property_types"] = property_recommendations
        
        return recommendations
    except Exception as e:
        print(f"Error generating property type recommendations: {str(e)}")
        return {"error": str(e)}

# Helper functions for formatting report sections
def _format_executive_summary(user_profile: Dict[str, Any], suburb_recommendations: List[Dict], 
                             property_recommendations: List[Dict], analysis_results: Dict) -> str:
    """
    Format the executive summary section of the recommendation report.
    
    Args:
        user_profile: User profile information
        suburb_recommendations: List of recommended suburbs
        property_recommendations: List of recommended property types
        analysis_results: Analysis results from data_analysis agent
        
    Returns:
        Formatted executive summary as string
    """
    # Get user details for personalization
    budget = user_profile.get('budget', 0)
    budget_str = f"${budget:,.2f}" if budget else "unspecified budget"
    
    # Get top suburb recommendation if available
    top_suburb = suburb_recommendations[0]['suburb'] if suburb_recommendations else "NSW"
    
    # Get top property type if available
    top_property = property_recommendations[0]['property_type'] if property_recommendations else "property"
    
    # Format the executive summary
    summary = f"""# NSW Housing Market Recommendation Report

## Executive Summary

Based on your profile and our analysis of the NSW housing market, we've identified optimal property opportunities aligned with your {budget_str} budget.

**Top Recommendation:** {top_property.title()} in {top_suburb}

Our analysis shows this combination offers the best match for your requirements, balancing affordability, growth potential, and your personal preferences. The following report provides detailed insights into recommended suburbs, property types, market trends, and important considerations to help you make an informed decision.
"""
    
    # Add key insights if available
    if 'key_insights' in analysis_results and analysis_results['key_insights']:
        summary += "\n**Key Market Insights:**\n"
        for insight in analysis_results['key_insights'][:3]:  # Top 3 insights
            summary += f"- {insight}\n"
    
    return summary

def _format_suburb_recommendations(suburb_recommendations: List[Dict], analysis_results: Dict) -> str:
    """
    Format the suburb recommendations section of the report.
    
    Args:
        suburb_recommendations: List of recommended suburbs
        analysis_results: Analysis results from data_analysis agent
        
    Returns:
        Formatted suburb recommendations as string
    """
    if not suburb_recommendations:
        return "## Recommended Suburbs\n\nNo specific suburb recommendations could be generated based on the available data."
    
    section = "## Recommended Suburbs\n\n"
    section += "The following suburbs in NSW are most aligned with your requirements:\n\n"
    
    # Add each recommended suburb with details
    for i, rec in enumerate(suburb_recommendations[:5]):  # Top 5 suburbs
        suburb = rec['suburb']
        section += f"### {i+1}. {suburb}\n\n"
        
        # Add justifications if available
        if rec['justifications']:
            section += "**Why this suburb:**\n"
            for j in rec['justifications']:
                section += f"- {j}\n"
            section += "\n"
        
        # Add recommended property types for this suburb
        if rec['recommended_property_types']:
            section += "**Recommended property types:** "
            section += ", ".join(p.title() for p in rec['recommended_property_types'])
            section += "\n\n"
        
        # Add warnings if any
        if rec['warnings']:
            section += "**Important considerations:**\n"
            for w in rec['warnings']:
                section += f"- {w}\n"
            section += "\n"
    
    return section

def _format_property_recommendations(property_recommendations: List[Dict]) -> str:
    """
    Format the property type recommendations section of the report.
    
    Args:
        property_recommendations: List of recommended property types
        
    Returns:
        Formatted property type recommendations as string
    """
    if not property_recommendations:
        return "## Recommended Property Types\n\nNo specific property type recommendations could be generated based on the available data."
    
    section = "## Recommended Property Types\n\n"
    section += "Based on your profile and market analysis, these property types are most suitable for your needs:\n\n"
    
    # Add each recommended property type with details
    for i, rec in enumerate(property_recommendations[:3]):  # Top 3 property types
        prop_type = rec['property_type'].title()
        median_price = rec['median_price']
        section += f"### {i+1}. {prop_type}\n\n"
        section += f"**Median price:** ${median_price:,.2f}\n\n"
        
        # Add distribution percentage if available
        if rec['distribution_percentage']:
            section += f"**Market share:** {rec['distribution_percentage']:.1f}% of NSW properties\n\n"
        
        # Add justifications if available
        if rec['justifications']:
            section += "**Why this property type:**\n"
            for j in rec['justifications']:
                section += f"- {j}\n"
            section += "\n"
    
    return section

def _format_market_trends(analysis_results: Dict) -> str:
    """
    Format the market trends section of the report.
    
    Args:
        analysis_results: Analysis results from data_analysis agent
        
    Returns:
        Formatted market trends as string
    """
    price_trends = analysis_results.get('price_trends', {})
    location_analysis = analysis_results.get('location_analysis', {})
    
    section = "## NSW Market Trends\n\n"
    
    # Overall market summary
    if 'overall_stats' in price_trends:
        overall = price_trends['overall_stats']
        median_price = overall.get('median_price', 0)
        yoy_change = overall.get('yoy_change', 0)
        
        section += f"The NSW housing market currently has a median price of ${median_price:,.2f} "
        if yoy_change > 0:
            section += f"with a {yoy_change:.1f}% increase over the past year.\n\n"
        elif yoy_change < 0:
            section += f"with a {abs(yoy_change):.1f}% decrease over the past year.\n\n"
        else:
            section += "with stable prices over the past year.\n\n"
    
    # Growth areas
    if 'growth_by_locality' in location_analysis and 'fastest_growing' in location_analysis['growth_by_locality']:
        fast_growing = location_analysis['growth_by_locality']['fastest_growing']
        if fast_growing:
            section += "**Fastest growing areas:**\n"
            for i, (locality, growth) in enumerate(list(fast_growing.items())[:5]):
                section += f"- {locality}: {growth:.1f}% growth\n"
            section += "\n"
    
    # Price trends by property type
    if 'price_by_property_type' in price_trends:
        section += "**Price trends by property type:**\n"
        for prop_type, stats in price_trends['price_by_property_type'].items():
            median = stats.get('median', 0)
            growth = stats.get('growth_rate', 0)
            section += f"- {prop_type.title()}: ${median:,.2f} median price, {growth:.1f}% annual change\n"
        section += "\n"
    
    return section

def _format_risk_factors(market_anomalies: List[Dict]) -> str:
    """
    Format the risk factors section of the report.
    
    Args:
        market_anomalies: List of detected market anomalies
        
    Returns:
        Formatted risk factors as string
    """
    section = "## Risk Factors and Considerations\n\n"
    
    if not market_anomalies:
        section += "Our analysis did not detect any significant market anomalies or risk factors at this time.\n"
        return section
    
    section += "When making your property decision, consider these identified market risks:\n\n"
    
    # Group anomalies by severity
    high_severity = [a for a in market_anomalies if a.get('severity') == 'High']
    medium_severity = [a for a in market_anomalies if a.get('severity') == 'Medium']
    
    # Add high severity anomalies
    if high_severity:
        section += "### High Priority Considerations\n\n"
        for anomaly in high_severity:
            section += f"- **{anomaly.get('type')}:** {anomaly.get('description')}\n"
        section += "\n"
    
    # Add medium severity anomalies
    if medium_severity:
        section += "### Additional Considerations\n\n"
        for anomaly in medium_severity:
            section += f"- **{anomaly.get('type')}:** {anomaly.get('description')}\n"
        section += "\n"
    
    return section

def _format_next_steps() -> str:
    """
    Format the next steps section of the report.
    
    Returns:
        Formatted next steps as string
    """
    return """## Next Steps

To proceed with your property search based on these recommendations:

1. **Visit recommended suburbs** to get a feel for the area and community
2. **Research local amenities** such as schools, transport, and shopping centers
3. **Contact local real estate agents** specializing in your preferred suburbs
4. **Attend property inspections** for your recommended property types
5. **Consider a pre-approval** for financing to strengthen your position when making offers
6. **Consult with professionals** including conveyancers and building inspectors before finalizing any purchase

Remember that the NSW property market can change rapidly, so it's advisable to make decisions based on the most current data available.
"""

def generate_recommendation_report(user_profile: Dict[str, Any], callback_context: CallbackContext) -> str:
    """
    Generate a comprehensive recommendation report based on all analysis results.
    This is the main function that produces the user-facing output.
    
    Args:
        user_profile: Dictionary containing user profile information
        callback_context: The callback context containing state information
    
    Returns:
        A formatted string containing the complete recommendation report
    """
    try:
        # First, ensure we have all the necessary recommendations
        # Generate suburb recommendations if not already done
        if "recommendations" not in callback_context.state or "suburbs" not in callback_context.state["recommendations"]:
            generate_suburb_recommendations(user_profile, callback_context)
            
        # Generate property type recommendations if not already done
        if "recommendations" not in callback_context.state or "property_types" not in callback_context.state["recommendations"]:
            generate_property_recommendations(user_profile, callback_context)
        
        # Get analysis results
        analysis_results = get_analysis_results(callback_context)
        
        # Get recommendations from state
        recommendations = callback_context.state.get("recommendations", {})
        suburb_recommendations = recommendations.get("suburbs", [])
        property_type_recommendations = recommendations.get("property_types", [])
        
        # Get market insights
        price_trends = analysis_results.get('price_trends', {})
        location_analysis = analysis_results.get('location_analysis', {})
        market_anomalies = analysis_results.get('market_anomalies', [])
        
        # Format the report sections
        report_sections = []
        
        # 1. Executive Summary
        exec_summary = _format_executive_summary(user_profile, suburb_recommendations, property_type_recommendations, analysis_results)
        report_sections.append(exec_summary)
        
        # 2. Suburb Recommendations
        suburb_section = _format_suburb_recommendations(suburb_recommendations, analysis_results)
        report_sections.append(suburb_section)
        
        # 3. Property Type Recommendations
        property_section = _format_property_recommendations(property_type_recommendations)
        report_sections.append(property_section)
        
        # 4. Market Trends
        trends_section = _format_market_trends(analysis_results)
        report_sections.append(trends_section)
        
        # 5. Risk Factors
        risk_section = _format_risk_factors(market_anomalies)
        report_sections.append(risk_section)
        
        # 6. Next Steps
        next_steps = _format_next_steps()
        report_sections.append(next_steps)
        
        # Combine all sections into the final report
        final_report = "\n\n".join(report_sections)
        
        # Store the report in state
        callback_context.state["recommendation_report"] = final_report
        
        return final_report
    except Exception as e:
        print(f"Error generating recommendation report: {str(e)}")
        return f"Error generating recommendation report: {str(e)}"


def create_final_recommendation(callback_context: CallbackContext) -> str:
    """
    Wrapper function to be called by the root agent to generate the final recommendation report.
    This function ensures that all necessary data is available and generates the complete report.
    
    Args:
        callback_context: The callback context containing state information
        
    Returns:
        The formatted recommendation report as a string
    """
    try:
        # Get user profile from state
        user_profile = callback_context.state.get("user_profile", {})
        
        if not user_profile:
            return "Error: No user profile found in state. Please collect user information first."
        
        # Check if analysis results are available
        analysis_results = callback_context.state.get("analysis_results", {})
        if not analysis_results:
            return "Error: No analysis results found. Please run the data analysis agent first."
        
        # Generate the recommendation report
        report = generate_recommendation_report(user_profile, callback_context)
        
        # Return the report
        return report
    except Exception as e:
        error_msg = f"Error creating final recommendation: {str(e)}"
        print(error_msg)
        return error_msg
