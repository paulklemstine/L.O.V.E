#!/bin/bash
# Auto-increment build number and deploy to Firebase
VERSION_FILE="public/version.json"
BUILD=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$VERSION_FILE')).build)")
NEW_BUILD=$((BUILD + 1))
echo "{\"build\": $NEW_BUILD}" > "$VERSION_FILE"
echo "🚀 Deploying build #$NEW_BUILD..."
npx firebase deploy --only hosting
echo "✅ Build #$NEW_BUILD deployed."
