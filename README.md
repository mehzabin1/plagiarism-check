# Plagiarism Checker in Python

A lightweight Python project that detects plagiarism between text documents using the **Cosine Similarity** algorithm.

## Overview

This project demonstrates a simple approach to plagiarism detection by comparing the similarity between text files.

Since computers process numerical data more efficiently than plain text, each document is first converted into a numerical vector. Once the text has been vectorized, cosine similarity is used to measure how closely two documents resemble each other.

The repository provides a beginner-friendly implementation of this concept using Python.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/mehzabin1/plagiarism-check.git
```

Or download the project as a ZIP file and extract it.

## Prerequisites

Install the required Python packages before running the application:

```bash
pip install -r requirements.txt
```

## Usage

Place all the text files you want to compare inside the project directory. Each document must have a **.txt** extension.

Run the application using:

```bash
cd plagiarism-check
python app.py
```

The program will automatically scan all `.txt` files in the directory, calculate pairwise similarity scores, and display the plagiarism percentage between each document pair.

## Customization

You can modify or extend this project to support additional file formats, preprocessing techniques, or alternative similarity algorithms based on your requirements.

## Need Help?

If you encounter any problems while setting up or running the project, feel free to open an issue in this repository.

## Contributions

Contributions are always welcome. If you have ideas for improvements, bug fixes, or new features, submit a pull request for review.

## Support

If you find this project helpful, consider giving the repository a ⭐ to support its development and help others discover it.
