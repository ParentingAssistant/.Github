#!/usr/bin/env python3
"""Test script for Step 37 - Pantry API endpoints"""

import requests
import json
import time
from datetime import datetime

# API configuration
BASE_URL = "http://localhost:8000"
# BASE_URL = "https://parenting-assistant-platform-39b47.ondigitalocean.app"  # Production

# Test user credentials (you'll need to update these)
TEST_USER_ID = 1
TEST_JWT = None  # We'll need to get this from auth

def test_auth():
    """Test authentication to get a JWT token"""
    print("\n1️⃣ Testing Authentication...")

    # For testing, we'll use a mock endpoint or existing token
    # In production, you'd use the Apple Sign In flow
    print("   ⚠️  Using mock authentication - update TEST_JWT manually if needed")
    return None

def test_profile_get(jwt):
    """Test fetching user profile with pantry"""
    print("\n2️⃣ Testing GET /v1/memory/profile...")

    headers = {}
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"

    try:
        response = requests.get(
            f"{BASE_URL}/v1/memory/profile",
            headers=headers
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("   ✅ Profile fetched successfully")

            # Check pantry field
            pantry = data.get("pantry", {})
            if pantry:
                print(f"   📦 Pantry items: {len(pantry.get('items', []))}")
                for item in pantry.get('items', [])[:3]:
                    print(f"      - {item.get('name')}: {item.get('quantity')} {item.get('unit', '')}")
            else:
                print("   📦 Pantry is empty")

            return data
        else:
            print(f"   ❌ Error: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return None

def test_pantry_update(jwt):
    """Test updating pantry in profile"""
    print("\n3️⃣ Testing PATCH /v1/memory/profile (pantry update)...")

    # Sample pantry data
    pantry_data = {
        "pantry": {
            "items": [
                {"name": "pasta", "quantity": 2, "unit": "boxes"},
                {"name": "olive oil", "quantity": 1, "unit": "bottle"},
                {"name": "tomatoes", "quantity": 5, "unit": "pieces"},
                {"name": "chicken breast", "quantity": 2, "unit": "pounds"},
                {"name": "milk", "quantity": 1, "unit": "gallon"},
                {"name": "eggs", "quantity": 12, "unit": "pieces"},
                {"name": "rice", "quantity": 1, "unit": "bag"},
                {"name": "onions", "quantity": 3, "unit": "pieces"}
            ],
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
    }

    headers = {"Content-Type": "application/json"}
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"

    try:
        response = requests.patch(
            f"{BASE_URL}/v1/memory/profile",
            json=pantry_data,
            headers=headers
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("   ✅ Pantry updated successfully")

            pantry = data.get("pantry", {})
            print(f"   📦 Updated pantry has {len(pantry.get('items', []))} items")

            return True
        else:
            print(f"   ❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return False

def test_meal_planning_with_pantry(jwt, user_id):
    """Test meal planning with pantry integration"""
    print("\n4️⃣ Testing POST /v1/assist/stream (meal planning with pantry)...")

    request_data = {
        "user_id": user_id,
        "goal": "Plan 3 easy dinners for this week",
        "mode": "meals"
    }

    headers = {"Content-Type": "application/json"}
    if jwt:
        headers["Authorization"] = f"Bearer {jwt}"

    try:
        print("   Sending meal planning request...")
        response = requests.post(
            f"{BASE_URL}/v1/assist/stream",
            json=request_data,
            headers=headers,
            stream=True
        )

        print(f"   Status: {response.status_code}")

        if response.status_code == 200:
            print("   Streaming response:")

            events = []
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    try:
                        event = json.loads(line_str)
                        events.append(event)

                        event_type = event.get("event")
                        if event_type == "token":
                            print(f"     💭 {event.get('data')}")
                        elif event_type == "final":
                            data = event.get("data", {})
                            meta = data.get("meta", {})

                            print("\n   ✅ Meal plan generated!")

                            # Check for grocery lists
                            grocery = meta.get("grocery")
                            missing_grocery = meta.get("missing_grocery")

                            if grocery:
                                total_items = sum(len(items) for items in grocery.get("aisles", {}).values())
                                print(f"   🛒 Full grocery list: {total_items} items")

                            if missing_grocery:
                                missing_items = sum(len(items) for items in missing_grocery.get("aisles", {}).values())
                                filtered_count = missing_grocery.get("_meta", {}).get("items_removed_count", 0)
                                print(f"   🛍️ Missing items only: {missing_items} items")
                                print(f"   ✅ Filtered out {filtered_count} items already in pantry")

                                # Show what was filtered
                                items_removed = missing_grocery.get("_meta", {}).get("items_removed", [])
                                if items_removed:
                                    print(f"   📦 Items you already have: {', '.join(items_removed[:5])}")

                        elif event_type == "error":
                            print(f"   ❌ Error: {event.get('data')}")

                    except json.JSONDecodeError:
                        print(f"     Raw: {line_str}")

            return events
        else:
            print(f"   ❌ Error: {response.text}")
            return None

    except Exception as e:
        print(f"   ❌ Request failed: {e}")
        return None

def test_direct_pantry_service():
    """Test the pantry service directly (if running locally)"""
    print("\n5️⃣ Testing pantry service logic (local only)...")

    try:
        import sys
        sys.path.append('/Users/ahmedm/Developer/ParentingAssistant/parenting-assistant-platform/gateway')

        from gateway.services.pantry_groceries import compute_missing_items

        pantry = {
            "items": [
                {"name": "pasta", "quantity": 2, "unit": "boxes"},
                {"name": "tomatoes", "quantity": 5, "unit": "pieces"},
                {"name": "chicken breast", "quantity": 2, "unit": "pounds"},
            ]
        }

        grocery_list = {
            "aisles": {
                "produce": ["tomato", "lettuce", "onions"],
                "meat": ["chicken breasts", "ground beef"],
                "pantry": ["pasta", "rice", "olive oil"]
            }
        }

        result = compute_missing_items(pantry, grocery_list)

        print("   ✅ Direct service test successful")
        print(f"   📦 Filtered out: {result.get('_meta', {}).get('items_removed')}")

        return True
    except ImportError:
        print("   ⚠️  Cannot test directly - service not available locally")
        return False
    except Exception as e:
        print(f"   ❌ Service test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("=" * 60)
    print("STEP 37 - PANTRY API TESTS")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print(f"Test User ID: {TEST_USER_ID}")

    # Get JWT token (you may need to set this manually)
    jwt = TEST_JWT or test_auth()

    if not jwt:
        print("\n⚠️  No JWT token available. Some tests may fail.")
        print("   Set TEST_JWT variable with a valid token to test authenticated endpoints")

    # Test profile fetch
    profile = test_profile_get(jwt)

    # Test pantry update
    if profile is not None:
        updated = test_pantry_update(jwt)

        if updated:
            # Test meal planning with pantry
            test_meal_planning_with_pantry(jwt, TEST_USER_ID)

    # Test direct service
    test_direct_pantry_service()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ Tests completed. Check results above.")
    print("\nTo fully test:")
    print("1. Start backend with: cd parenting-assistant-platform && docker compose up")
    print("2. Get a valid JWT token from Apple Sign In flow")
    print("3. Update TEST_JWT in this script")
    print("4. Run this script again")

if __name__ == "__main__":
    main()