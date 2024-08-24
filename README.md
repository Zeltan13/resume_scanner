# 📄 Resume Scanner: A Fun NLP Project

Welcome to my **Resume Scanner** project! This is a personal NLP (Natural Language Processing) project I built just for fun. The idea is simple but powerful: create an AI-powered tool that scans a resume and predicts the most suitable job category based on the content.

## 🚀 Project Overview

Ever wondered what job your resume screams out for? This project leverages machine learning to analyze resumes and categorize them into job roles like "Data Scientist," "Web Developer," and more. It's a great way to see if your resume aligns with your career aspirations!

### 🛠️ Features

- **PDF & Text File Support**: The scanner can handle both PDF and text resumes. Just upload your file, and let the magic happen!
- **Machine Learning Model**: Powered by a Logistic Regression model trained on a dataset of various resumes and job categories.
- **NLP Techniques**: Utilizes TF-IDF vectorization to convert resume text into features, and classic text preprocessing techniques to clean the data.
- **Interactive Web Interface**: Built with Flask, you can easily interact with the model through a simple web app.

## 📦 Technologies Used

- **Python**: The core language used to build the project.
- **Flask**: A lightweight web framework to create the interactive interface.
- **scikit-learn**: For the machine learning model and text vectorization.
- **pdfminer.six**: To extract text from PDF resumes.
- **nltk**: For text preprocessing, including stopword removal and tokenization.

## 🎯 How It Works

1. **Upload Your Resume**: You can upload your resume in either PDF or text format.
2. **Text Extraction & Preprocessing**: If it's a PDF, the app extracts the text. The text is then cleaned by removing unnecessary characters and stopwords.
3. **Feature Extraction**: The cleaned text is converted into numerical features using TF-IDF vectorization.
4. **Prediction**: The machine learning model predicts the job category that best matches the content of your resume.
5. **Result**: The predicted job category is displayed on the web page.

## 💡 Inspiration

This project was inspired by the idea of combining my love for NLP with a practical application that many people could find useful. Plus, it was a fun way to dive deeper into how text data can be used in machine learning.

## 🛠️ Getting Started

### Prerequisites

- Python 3.x
