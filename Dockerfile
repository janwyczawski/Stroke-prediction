FROM python:3.12
COPY app.py /app.py
COPY logistic_regression_tuned.pkl /logistic_regression_tuned.pkl
RUN pip install flask joblib imbalanced-learn pandas
EXPOSE 5000
CMD ["python", "/app.py"]