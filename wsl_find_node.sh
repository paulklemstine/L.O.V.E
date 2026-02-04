#!/bin/bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" > /dev/null 2>&1
nvm use 22 > /dev/null 2>&1
which node
