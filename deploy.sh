#!/bin/bash
# Auto-increment build number and deploy to Firebase
cd "$(dirname "$(realpath "$0")")"
VERSION_FILE="public/version.json"
BUILD=$(python3 -c "import json; print(json.load(open('$VERSION_FILE'))['build'])")
NEW_BUILD=$((BUILD + 1))
python3 -c "import json; json.dump({'build': $NEW_BUILD}, open('$VERSION_FILE', 'w'))"
# Stamp build number into index.html
sed -i "s/build #<span id=\"build-version\">[0-9]*<\/span>/build #<span id=\"build-version\">$NEW_BUILD<\/span>/" public/index.html
echo "Deploying build #$NEW_BUILD..."
npx firebase deploy --only hosting
echo "Build #$NEW_BUILD deployed."
