#!/bin/bash
flake8 ./netharn --count --select=E9,F63,F7,F82 --show-source --statistics
