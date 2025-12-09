#!/bin/bash
cd /d/DEVOPS-PROJECT-main
git add .
git commit -m "feat: Zero-configuration Docker startup with automatic model training

Changes:
- Create entrypoint.sh for automatic model training on startup
- Update Dockerfile to use ENTRYPOINT with the startup script
- Add jupyter and jupyterlab to requirements.txt
- Update docker-compose.yml: increase health check start_period to 90s
- Remove pip install from jupyter service command
- Enhance .dockerignore with comprehensive exclusions

Benefits:
- docker-compose up handles everything automatically
- No manual training commands needed
- Model training on startup eliminates need for pre-baked models
- Simplified deployment process"
git push origin master
