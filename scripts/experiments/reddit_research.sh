#!/bin/bash
USER="Ron_p_wilder"
# List of subreddits to check
SUBS=("artphotography" "shibari" "FineArtNudes" "boudoir" "Photography" "analog" "leica" "hasselblad")

for sub in "${SUBS[@]}"; do
    echo "Searching in r/$sub..."
    # Search for user posts in specific subreddits via Google search (less likely to be blocked)
    curl -s -A "Mozilla/5.0" "https://www.google.com/search?q=site:reddit.com/r/$sub+%22$USER%22" | grep -o "reddit.com/r/$sub/comments/[^/]*" | head -n 3
done
