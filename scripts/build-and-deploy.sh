#! /bin/bash

./scripts/build-all.sh
cd ./online-playground
firebase deploy
