"""
Tools for the data analysis agent to analyze NSW housing market data.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from vertexai.generative_models import CallbackContext

# Core data loading functions
def load_processed_data(callback_context: CallbackContext) -> pd.DataFrame:
    """
    Load the processed housing data CSV file.
    
    Args:
        callback_context: The agent's callback context containing state
        
    Returns:
        DataFrame containing the processed housing data
    """
    data_path = callback_context.state.get("processed_data_path")
    if not data_path or not os.path.exists(data_path):
        # Try to find the file if path not in state
        data_dir = Path.cwd() / "data"
        data_path = str(data_dir / "processed-housing-data.csv")
    
    try:
        df = pd.read_csv(data_path)
        # Ensure contract date is proper datetime format
        if 'contract_date' in df.columns:
            df['contract_date'] = pd.to_datetime(df['contract_date'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Price analysis functions
def analyze_price_trends(data: pd.DataFrame, callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Analyze price trends in the housing data by location, property type, and time period.
    
    Args:
        data: DataFrame containing housing data
        callback_context: The agent's callback context
        
    Returns:
        Dictionary containing price trend analysis results
    """
    results = {}
    
    try:
        # Ensure we have the necessary columns
        if 'purchase_price' not in data.columns or data.empty:
            return {'error': 'Missing price data'}
            
        # Overall price statistics
        results['overall_stats'] = {
            'median_price': data['purchase_price'].median(),
            'mean_price': data['purchase_price'].mean(),
            'min_price': data['purchase_price'].min(),
            'max_price': data['purchase_price'].max(),
            'price_std': data['purchase_price'].std(),
        }
        
        # Price by locality (suburb)
        if 'locality' in data.columns:
            locality_prices = data.groupby('locality')['purchase_price'].agg(['median', 'mean', 'count'])
            locality_prices = locality_prices.sort_values('count', ascending=False)
            top_localities = locality_prices.head(20).to_dict('index')
            results['price_by_locality'] = top_localities
            
        # Price by property type
        if 'property_type' in data.columns:
            type_prices = data.groupby('property_type')['purchase_price'].agg(['median', 'mean', 'count'])
            results['price_by_property_type'] = type_prices.to_dict('index')
        
        # Time-based analysis if date column exists
        if 'contract_date' in data.columns:
            # Add year and month columns
            data['year'] = data['contract_date'].dt.year
            data['month'] = data['contract_date'].dt.month
            
            # Price trends by year
            yearly_trends = data.groupby('year')['purchase_price'].agg(['median', 'mean', 'count'])
            results['yearly_price_trends'] = yearly_trends.to_dict('index')
            
            # Price trends by month (recent year only if multiple years exist)
            recent_year = data['year'].max()
            recent_data = data[data['year'] == recent_year]
            monthly_trends = recent_data.groupby('month')['purchase_price'].agg(['median', 'mean', 'count'])
            results['monthly_price_trends'] = monthly_trends.to_dict('index')
        
        # Store results in state
        callback_context.state['price_trends'] = results
        
        return results
    except Exception as e:
        print(f"Error analyzing price trends: {str(e)}")
        return {'error': str(e)}

# Location analysis functions
def analyze_locations(data: pd.DataFrame, callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Analyze locations in the housing data to identify popular areas and their characteristics.
    
    Args:
        data: DataFrame containing housing data
        callback_context: The agent's callback context
        
    Returns:
        Dictionary containing location analysis results
    """
    results = {}
    
    try:
        # Check if we have necessary columns
        if 'locality' not in data.columns or data.empty:
            return {'error': 'Missing location data'}
            
        # Count properties by locality
        location_counts = data['locality'].value_counts().head(30).to_dict()
        results['top_localities_by_count'] = location_counts
        
        # Analyze price points by locality
        if 'purchase_price' in data.columns:
            # Get median prices by locality
            median_by_locality = data.groupby('locality')['purchase_price'].median().sort_values(ascending=False)
            results['top_expensive_localities'] = median_by_locality.head(20).to_dict()
            results['most_affordable_localities'] = median_by_locality.tail(20).to_dict()
            
            # Identify fast-growing areas if time data exists
            if 'contract_date' in data.columns and 'year' in data.columns:
                # Get the two most recent years
                years = sorted(data['year'].unique())
                if len(years) >= 2:
                    recent_year = years[-1]
                    previous_year = years[-2]
                    
                    # Calculate price growth between the two years
                    recent_prices = data[data['year'] == recent_year].groupby('locality')['purchase_price'].median()
                    prev_prices = data[data['year'] == previous_year].groupby('locality')['purchase_price'].median()
                    
                    # Only include localities present in both years
                    common_localities = set(recent_prices.index) & set(prev_prices.index)
                    growth_rates = {}
                    
                    for locality in common_localities:
                        recent = recent_prices[locality]
                        prev = prev_prices[locality]
                        if prev > 0:  # Avoid division by zero
                            growth = (recent - prev) / prev * 100
                            growth_rates[locality] = growth
                    
                    # Sort by growth rate
                    sorted_growth = {k: v for k, v in sorted(
                        growth_rates.items(), 
                        key=lambda item: item[1], 
                        reverse=True
                    )}
                    
                    results['growth_by_locality'] = {
                        'fastest_growing': dict(list(sorted_growth.items())[:15]),
                        'slowest_growing': dict(list(sorted_growth.items())[-15:])
                    }
        
        # Analyze by postcode if available
        if 'post_code' in data.columns:
            postcode_data = data.groupby('post_code').agg({
                'purchase_price': ['median', 'mean', 'count'],
                'locality': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
            })
            results['postcode_analysis'] = postcode_data.head(20).to_dict()
            
        # Store results in state
        callback_context.state['location_analysis'] = results
        
        return results
    except Exception as e:
        print(f"Error analyzing locations: {str(e)}")
        return {'error': str(e)}

# Property type analysis functions
def analyze_property_types(data: pd.DataFrame, callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Analyze property types in the housing data to identify trends and characteristics.
    
    Args:
        data: DataFrame containing housing data
        callback_context: The agent's callback context
        
    Returns:
        Dictionary containing property type analysis results
    """
    results = {}
    
    try:
        # Check if we have necessary columns
        if 'property_type' not in data.columns or data.empty:
            return {'error': 'Missing property type data'}
            
        # Count properties by type
        type_counts = data['property_type'].value_counts().to_dict()
        results['property_type_distribution'] = type_counts
        
        # Analyze price by property type
        if 'purchase_price' in data.columns:
            price_by_type = data.groupby('property_type')['purchase_price'].agg(['median', 'mean', 'min', 'max', 'count'])
            results['price_by_property_type'] = price_by_type.to_dict('index')
            
        # Analyze property types by locality if both columns exist
        if 'locality' in data.columns:
            # Find top property types for popular localities
            top_localities = data['locality'].value_counts().head(10).index.tolist()
            
            type_by_locality = {}
            for locality in top_localities:
                locality_data = data[data['locality'] == locality]
                type_by_locality[locality] = locality_data['property_type'].value_counts().to_dict()
                
            results['property_types_by_top_localities'] = type_by_locality
            
        # Analyze property type trends over time if date column exists
        if 'contract_date' in data.columns and 'year' in data.columns:
            # Property type distribution by year
            type_by_year = {}
            for year in data['year'].unique():
                year_data = data[data['year'] == year]
                type_by_year[int(year)] = year_data['property_type'].value_counts().to_dict()
                
            results['property_type_trends_by_year'] = type_by_year
            
        # Analyze primary purpose by property type if available
        if 'primary_purpose' in data.columns:
            purpose_by_type = {}
            for prop_type in data['property_type'].unique():
                type_data = data[data['property_type'] == prop_type]
                purpose_by_type[prop_type] = type_data['primary_purpose'].value_counts().to_dict()
                
            results['primary_purpose_by_property_type'] = purpose_by_type
        
        # Store results in state
        callback_context.state['property_type_analysis'] = results
        
        return results
    except Exception as e:
        print(f"Error analyzing property types: {str(e)}")
        return {'error': str(e)}

# Anomaly and risk detection functions
def detect_market_anomalies(data: pd.DataFrame, callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Detect anomalies and risks in the housing data such as unusual price movements or outliers.
    
    Args:
        data: DataFrame containing housing data
        callback_context: The agent's callback context
        
    Returns:
        Dictionary containing detected anomalies and risk factors
    """
    anomalies = []
    results = {}
    
    try:
        # Check if we have necessary columns
        if 'purchase_price' not in data.columns or data.empty:
            return {'error': 'Missing price data for anomaly detection'}
        
        # Calculate price statistics for outlier detection
        price_mean = data['purchase_price'].mean()
        price_std = data['purchase_price'].std()
        price_median = data['purchase_price'].median()
        
        # Identify global price outliers (3 standard deviations from mean)
        high_threshold = price_mean + 3 * price_std
        low_threshold = max(price_mean - 3 * price_std, 0)  # Ensure non-negative
        
        high_outliers = data[data['purchase_price'] > high_threshold]
        low_outliers = data[data['purchase_price'] < low_threshold]
        
        results['global_outliers'] = {
            'high_outlier_count': len(high_outliers),
            'low_outlier_count': len(low_outliers),
            'high_threshold': high_threshold,
            'low_threshold': low_threshold
        }
        
        if len(high_outliers) > 0:
            anomalies.append({
                'type': 'High price outliers',
                'description': f'Found {len(high_outliers)} properties with unusually high prices (>{high_threshold:,.2f})',
                'severity': 'Medium'
            })
        
        # Analyze by locality if available
        if 'locality' in data.columns:
            # Find localities with significant price variance
            loc_stats = data.groupby('locality')['purchase_price'].agg(['mean', 'std', 'median', 'count'])
            loc_stats = loc_stats[loc_stats['count'] >= 5]  # Only consider localities with sufficient data
            
            # Calculate coefficient of variation (CV) to identify high variance areas
            loc_stats['cv'] = loc_stats['std'] / loc_stats['mean'] * 100
            high_variance_locs = loc_stats[loc_stats['cv'] > 50].sort_values('cv', ascending=False)
            
            if len(high_variance_locs) > 0:
                results['high_variance_localities'] = high_variance_locs.head(10).to_dict('index')
                anomalies.append({
                    'type': 'Price volatility',
                    'description': f'High price variance detected in {len(high_variance_locs)} localities',
                    'severity': 'Medium',
                    'affected_localities': high_variance_locs.head(5).index.tolist()
                })
        
        # Time-based anomaly detection if date column exists
        if 'contract_date' in data.columns and 'year' in data.columns and 'month' in data.columns:
            # Check for unusual monthly price movements
            monthly_prices = data.groupby(['year', 'month'])['purchase_price'].median()
            
            if len(monthly_prices) > 2:
                # Calculate month-to-month price changes
                monthly_prices = monthly_prices.reset_index()
                monthly_prices['prev_price'] = monthly_prices['purchase_price'].shift(1)
                monthly_prices['price_change_pct'] = (monthly_prices['purchase_price'] - monthly_prices['prev_price']) / monthly_prices['prev_price'] * 100
                
                # Identify months with significant price changes (>10% month-over-month)
                significant_changes = monthly_prices[abs(monthly_prices['price_change_pct']) > 10].dropna()
                
                if len(significant_changes) > 0:
                    results['significant_monthly_changes'] = significant_changes[['year', 'month', 'price_change_pct']].to_dict('records')
                    
                    # Add to anomalies list
                    for _, row in significant_changes.iterrows():
                        direction = 'increase' if row['price_change_pct'] > 0 else 'decrease'
                        anomalies.append({
                            'type': f'Significant price {direction}',
                            'description': f'{abs(row["price_change_pct"]):.1f}% {direction} in {int(row["month"])}/{int(row["year"])}',
                            'severity': 'High' if abs(row['price_change_pct']) > 20 else 'Medium'
                        })
        
        # Store results in state
        results['anomalies'] = anomalies
        callback_context.state['market_anomalies'] = anomalies
        
        return results
    except Exception as e:
        print(f"Error detecting anomalies: {str(e)}")
        return {'error': str(e), 'anomalies': []}

# User profile matching functions
def match_properties_to_user_profile(data: pd.DataFrame, user_profile: Dict[str, Any], callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Match properties to user profile based on budget, location preferences, and other criteria.
    
    Args:
        data: DataFrame containing housing data
        user_profile: Dictionary containing user profile information
        callback_context: The agent's callback context
        
    Returns:
        Dictionary containing matched properties and recommendations
    """
    results = {}
    
    try:
        # Check if we have necessary columns and data
        if data.empty or 'purchase_price' not in data.columns:
            return {'error': 'Missing required data for matching'}
        
        # Extract user profile details
        user_budget = user_profile.get('budget', 0)
        preferred_locations = user_profile.get('preferred_locations', [])
        preferred_property_types = user_profile.get('preferred_property_types', [])
        family_size = user_profile.get('family_size', 1)
        income = user_profile.get('income', 0)
        
        # Basic filtering based on budget
        budget_filter = data['purchase_price'] <= user_budget * 1.1  # Allow 10% over budget for flexibility
        budget_matches = data[budget_filter]
        
        results['budget_match_count'] = len(budget_matches)
        results['budget_match_percentage'] = len(budget_matches) / len(data) * 100 if len(data) > 0 else 0
        
        # Location preference matching
        location_matches = pd.DataFrame()
        if preferred_locations and 'locality' in data.columns:
            location_filter = data['locality'].isin(preferred_locations)
            location_matches = data[location_filter]
            
            results['location_match_count'] = len(location_matches)
            
            # Matches within both budget and preferred locations
            budget_location_matches = budget_matches[budget_matches['locality'].isin(preferred_locations)]
            results['budget_location_match_count'] = len(budget_location_matches)
            
        # Property type matching
        type_matches = pd.DataFrame()
        if preferred_property_types and 'property_type' in data.columns:
            type_filter = data['property_type'].isin(preferred_property_types)
            type_matches = data[type_filter]
            
            results['property_type_match_count'] = len(type_matches)
            
            # Triple matches: budget + location + property type
            if len(location_matches) > 0:
                triple_matches = budget_matches[
                    (budget_matches['locality'].isin(preferred_locations)) & 
                    (budget_matches['property_type'].isin(preferred_property_types))
                ]
                results['ideal_match_count'] = len(triple_matches)
        
        # Calculate affordability metrics if income is provided
        if income > 0:
            # Calculate price-to-income ratio (ideally 3-5x annual income)
            median_price = data['purchase_price'].median()
            price_to_income = median_price / income
            results['price_to_income_ratio'] = price_to_income
            
            # Affordability assessment
            if price_to_income <= 3:
                affordability = "Very affordable"
            elif price_to_income <= 5:
                affordability = "Affordable"
            elif price_to_income <= 7:
                affordability = "Moderately expensive"
            else:
                affordability = "Expensive"
                
            results['affordability_assessment'] = affordability
        
        # Find the best locality matches based on user criteria
        best_localities = []
        if 'locality' in data.columns:
            # Group by locality and calculate median prices
            locality_stats = data.groupby('locality').agg({'purchase_price': 'median'}).reset_index()
            
            # Filter localities by budget
            affordable_localities = locality_stats[locality_stats['purchase_price'] <= user_budget]
            
            # Sort by how close they are to the user's budget (making best use of budget)
            affordable_localities['budget_utilization'] = affordable_localities['purchase_price'] / user_budget * 100
            best_budget_localities = affordable_localities.sort_values('budget_utilization', ascending=False)
            
            # Take top localities
            best_localities = best_budget_localities.head(10)['locality'].tolist()
            results['recommended_localities'] = best_localities
        
        # Store results in state
        callback_context.state['user_matching_results'] = results
        
        return results
    except Exception as e:
        print(f"Error matching user profile: {str(e)}")
        return {'error': str(e)}

# Comprehensive analysis function
def run_comprehensive_analysis(user_profile: Dict[str, Any], callback_context: CallbackContext) -> Dict[str, Any]:
    """
    Run a comprehensive analysis of housing data based on user profile and preferences.
    This function calls all other analysis functions and consolidates the results.
    
    Args:
        user_profile: Dictionary containing user profile information
        callback_context: The agent's callback context
        
    Returns:
        Dictionary containing consolidated analysis results
    """
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_profile': user_profile
    }
    
    try:
        # Load the processed data
        data = load_processed_data(callback_context)
        if data.empty:
            return {'error': 'Failed to load processed housing data'}
        
        print(f"Loaded processed housing data with {len(data)} records")
        
        # Run all analysis functions
        print("Running price trend analysis...")
        price_trends = analyze_price_trends(data, callback_context)
        results['price_trends'] = price_trends
        
        print("Running location analysis...")
        location_analysis = analyze_locations(data, callback_context)
        results['location_analysis'] = location_analysis
        
        print("Running property type analysis...")
        property_type_analysis = analyze_property_types(data, callback_context)
        results['property_type_analysis'] = property_type_analysis
        
        print("Running anomaly detection...")
        anomalies = detect_market_anomalies(data, callback_context)
        results['market_anomalies'] = anomalies.get('anomalies', [])
        
        # User profile matching (if profile is provided)
        if user_profile:
            print("Matching properties to user profile...")
            user_matches = match_properties_to_user_profile(data, user_profile, callback_context)
            results['user_matches'] = user_matches
            
        # Generate key insights
        insights = []
        
        # Price trend insights
        if 'overall_stats' in price_trends:
            median_price = price_trends['overall_stats'].get('median_price')
            insights.append(f"The median property price in NSW is ${median_price:,.2f}")
        
        # Location insights
        if 'top_expensive_localities' in location_analysis and 'most_affordable_localities' in location_analysis:
            top_expensive = next(iter(location_analysis['top_expensive_localities'].items()), None)
            most_affordable = next(iter(location_analysis['most_affordable_localities'].items()), None)
            
            if top_expensive and most_affordable:
                insights.append(f"The most expensive area is {top_expensive[0]} while the most affordable is {most_affordable[0]}")
        
        if 'growth_by_locality' in location_analysis and 'fastest_growing' in location_analysis.get('growth_by_locality', {}):
            fastest_growing = next(iter(location_analysis['growth_by_locality']['fastest_growing'].items()), None)
            if fastest_growing:
                insights.append(f"{fastest_growing[0]} is showing the fastest price growth at {fastest_growing[1]:.1f}%")
        
        # User profile specific insights
        if 'user_matches' in results:
            if 'affordability_assessment' in results['user_matches']:
                insights.append(f"The market is {results['user_matches']['affordability_assessment'].lower()} based on your income")
            
            if 'recommended_localities' in results['user_matches'] and results['user_matches']['recommended_localities']:
                top_recommended = results['user_matches']['recommended_localities'][0]
                insights.append(f"{top_recommended} appears to be the best match for your budget and preferences")
        
        # Add anomaly insights
        if results['market_anomalies']:
            for anomaly in results['market_anomalies'][:2]:  # Top 2 anomalies only
                insights.append(f"Market risk: {anomaly['description']}")
        
        results['key_insights'] = insights
        
        # Store consolidated results in callback context
        callback_context.state['analysis_results'] = results
        
        return results
    except Exception as e:
        print(f"Error in comprehensive analysis: {str(e)}")
        return {'error': str(e)}
