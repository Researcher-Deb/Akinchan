"""
Test script to verify new simulation ID format
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_new_simulation():
    """Test creating a simulation with the new ID format"""
    
    print("="*60)
    print("Testing New Simulation ID Format")
    print("="*60)
    
    # Login first
    print("\n1. Logging in...")
    login_data = {
        "username": "user1",
        "password": "Cloud_Run-25"
    }
    
    response = requests.post(f"{BASE_URL}/api/login", json=login_data)
    if response.status_code != 200:
        print(f"❌ Login failed: {response.status_code}")
        print(response.text)
        return
    
    user_data = response.json()
    print(f"✅ Logged in as: {user_data['user']['username']}")
    print(f"   User ID: {user_data['user']['user_id']}")
    
    # Run a simulation
    print("\n2. Running simulation...")
    sim_data = {
        "trial_id": "NCT04521789",  # ImmunoX-101 trial
        "user_id": user_data['user']['user_id'],
        "username": user_data['user']['username']
    }
    
    response = requests.post(f"{BASE_URL}/api/simulate", json=sim_data)
    if response.status_code != 200:
        print(f"❌ Simulation failed: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    print(f"✅ Simulation completed!")
    print(f"\n   OLD FORMAT: 5ba85d64-62a7-4de1-a8ea-8764cfd4ec7e")
    print(f"   NEW FORMAT: {result['simulation_id']}")
    print(f"\n   Expected format: trial_name_####")
    print(f"   Example: study_of_immunox_101_1234")
    
    # Check format
    sim_id = result['simulation_id']
    if '_' in sim_id and sim_id.split('_')[-1].isdigit():
        print(f"\n   ✅ ID format is correct!")
        print(f"   - Contains underscores: ✓")
        print(f"   - Ends with numbers: ✓")
        print(f"   - Readable: ✓")
    else:
        print(f"\n   ⚠️ ID format might need adjustment")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)

if __name__ == "__main__":
    try:
        test_new_simulation()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
