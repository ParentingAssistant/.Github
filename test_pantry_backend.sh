#!/bin/bash

# Test script to verify backend pantry behavior

BASE_URL="https://parenting-assistant-platform-39b47.ondigitalocean.app"

echo "Testing Pantry Backend Issue"
echo "============================="

# You'll need to set this to a valid JWT token
JWT_TOKEN="${JWT_TOKEN:-your_jwt_here}"

if [ "$JWT_TOKEN" = "your_jwt_here" ]; then
    echo "Please set JWT_TOKEN environment variable with a valid token"
    echo "You can get one from the iOS app after signing in"
    exit 1
fi

echo ""
echo "1. Fetching current profile..."
curl -s -H "Authorization: Bearer $JWT_TOKEN" \
    "$BASE_URL/v1/memory/profile" | python3 -m json.tool | grep -A5 '"pantry"'

echo ""
echo "2. Updating pantry with test data..."
RESPONSE=$(curl -s -X PATCH \
    -H "Authorization: Bearer $JWT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
        "pantry": {
            "items": [
                {"name": "Test Item", "quantity": 2, "unit": "pieces"}
            ],
            "last_updated": "2025-11-28T02:00:00Z"
        }
    }' \
    "$BASE_URL/v1/memory/profile")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool | grep -A10 '"pantry"'

echo ""
echo "3. Fetching profile again to verify..."
curl -s -H "Authorization: Bearer $JWT_TOKEN" \
    "$BASE_URL/v1/memory/profile" | python3 -m json.tool | grep -A5 '"pantry"'

echo ""
echo "============================="
echo "If pantry.items is null in the response, it's a backend issue"