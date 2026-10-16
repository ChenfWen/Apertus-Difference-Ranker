FROM python:3.12-slim

WORKDIR /app
COPY src/submission_replay.py /app/submission_replay.py
COPY data/results/final_test_8b_summary.json /app/result-8b.json
COPY data/results/final_test_70b_summary.json /app/result-70b.json

ENTRYPOINT ["python", "/app/submission_replay.py", "/app/result-8b.json", "/app/result-70b.json"]
