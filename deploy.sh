#!/bin/bash
# Auto-increment build number and deploy to Firebase.
# Restored 2026-10-03 after commit d9bd66d8 deleted it; CLAUDE.md still documented
# it, and the build-number stamping it performs is still live in public/index.html.
#
# The original trusted public/version.json alone. Those two have since drifted
# (version.json said 91 while the page was stamped 94), which would have bumped
# the build to 92 and stamped the page BACKWARDS. Take the higher of the two.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

VERSION_FILE="public/version.json"
INDEX_FILE="public/index.html"

file_build=$(python3 -c "import json;print(json.load(open('$VERSION_FILE'))['build'])" 2>/dev/null || echo 0)
html_build=$(grep -o 'id="build-version">[0-9]*' "$INDEX_FILE" | grep -o '[0-9]*' | head -1 || echo 0)
html_build=${html_build:-0}

BUILD=$(( file_build > html_build ? file_build : html_build ))
NEW_BUILD=$((BUILD + 1))

python3 -c "import json; json.dump({'build': $NEW_BUILD}, open('$VERSION_FILE', 'w'))"
# Rewrite only the digits INSIDE the span, never a whole-tag pattern. The markup
# wraps between "build #<span" and "id="build-version"" onto two lines and uses
# CRLF endings, so the original single-line sed never matched — which is exactly
# why version.json and index.html had drifted. Matching the span contents alone
# is line-local and leaves the \r untouched.
sed -i "s/id=\"build-version\">[0-9]*/id=\"build-version\">$NEW_BUILD/" "$INDEX_FILE"

if [ "$file_build" != "$html_build" ]; then
    echo "Note: version.json ($file_build) and index.html ($html_build) had drifted; reconciled to $NEW_BUILD."
fi

echo "Deploying build #$NEW_BUILD..."
npx --yes firebase-tools deploy --only hosting
echo "Build #$NEW_BUILD deployed."