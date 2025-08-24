#!/usr/bin/env python3
"""
Sample Data Generator for Onboarding Drop-off Analyzer

This script generates realistic sample data to test the application.
Run this script to create a sample CSV file for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_users=1000, output_file='sample_onboarding_data.csv'):
    """
    Generate sample onboarding funnel data
    
    Args:
        n_users: Number of users to generate
        output_file: Output CSV filename
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define onboarding steps
    steps = [
        'signup',
        'email_verification', 
        'profile_setup',
        'preferences',
        'onboarding_complete'
    ]
    
    # User properties
    email_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com', 'protonmail.com']
    devices = ['mobile', 'desktop', 'tablet']
    locations = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN', 'BR', 'MX']
    
    # Base time for simulation
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    data = []
    
    print(f"Generating data for {n_users} users...")
    
    for user_id in range(1, n_users + 1):
        if user_id % 100 == 0:
            print(f"Generated {user_id} users...")
        
        # Random user properties
        email_domain = random.choice(email_domains)
        device = random.choice(devices)
        location = random.choice(locations)
        
        # Simulate funnel progression with realistic drop-offs
        current_step = 0
        user_start_time = base_time + timedelta(
            hours=random.randint(0, 24*30),  # Random time within a month
            minutes=random.randint(0, 60)
        )
        
        for step in steps:
            # Base completion probability (decreases as we go deeper)
            base_completion_prob = 0.95 - (current_step * 0.12)
            
            # Adjust probability based on user properties
            completion_prob = base_completion_prob
            
            # Mobile users struggle more with complex steps
            if device == 'mobile' and step in ['profile_setup', 'preferences']:
                completion_prob *= 0.75
            
            # Popular email domains have higher verification success
            if email_domain in ['gmail.com', 'outlook.com'] and step == 'email_verification':
                completion_prob *= 1.15
            
            # Location-based adjustments
            if location in ['US', 'UK', 'CA'] and step == 'profile_setup':
                completion_prob *= 1.1  # English-speaking countries do better
            
            # Time-based adjustments (weekend vs weekday)
            is_weekend = user_start_time.weekday() >= 5
            if is_weekend and step in ['profile_setup', 'preferences']:
                completion_prob *= 0.9  # Weekend users are less likely to complete complex steps
            
            # Determine if user completes this step
            completed = random.random() < completion_prob
            
            # Add some realistic timing between steps
            step_delay = random.randint(1, 10)  # 1-10 minutes between steps
            
            data.append({
                'user_id': f'user_{user_id:04d}',
                'step': step,
                'completed': completed,
                'email_domain': email_domain,
                'device': device,
                'location': location,
                'timestamp': user_start_time + timedelta(minutes=current_step * step_delay)
            })
            
            if not completed:
                break  # User dropped off at this step
            
            current_step += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Sample data generated successfully!")
    print(f"📁 File saved as: {output_file}")
    print(f"📊 Total records: {len(df)}")
    print(f"👥 Unique users: {df['user_id'].nunique()}")
    print(f"🔢 Steps: {df['step'].nunique()}")
    
    # Show some statistics
    print(f"\n📈 Funnel Summary:")
    for step in steps:
        step_data = df[df['step'] == step]
        total_users = step_data['user_id'].nunique()
        completed_users = step_data[step_data['completed']]['user_id'].nunique()
        completion_rate = (completed_users / total_users) * 100 if total_users > 0 else 0
        
        print(f"  {step:20} | {total_users:4d} users | {completion_rate:5.1f}% completion")
    
    return df

def create_small_sample(n_users=100, output_file='small_sample_data.csv'):
    """
    Create a smaller sample for quick testing
    """
    return generate_sample_data(n_users, output_file)

if __name__ == "__main__":
    print("🚀 Onboarding Drop-off Analyzer - Sample Data Generator")
    print("=" * 60)
    
    # Generate main sample
    df = generate_sample_data(1000, 'sample_onboarding_data.csv')
    
    # Also create a small sample for quick testing
    create_small_sample(100, 'small_sample_data.csv')
    
    print("\n🎯 You can now:")
    print("1. Use 'sample_onboarding_data.csv' for full analysis")
    print("2. Use 'small_sample_data.csv' for quick testing")
    print("3. Run 'streamlit run app.py' to start the analyzer")
    print("4. Upload either CSV file to test the application")
    
    print("\n💡 Tip: The sample data includes realistic patterns:")
    print("   - Mobile users struggle more with complex steps")
    print("   - Popular email domains have higher verification success")
    print("   - Weekend users are less likely to complete complex steps")
    print("   - Location-based completion rate variations")

