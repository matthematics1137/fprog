#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 00:11:56 2023

@author: somebody
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
import pdfkit
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import re
import weaviate
import PyPDF2
from PyPDF2 import PdfReader
from urllib.parse import quote

# Initialize the Weaviate client
client = weaviate.Client("http://localhost:8080")
time.sleep(10)

def save_to_weaviate(dest_folder, company_name):
    for root, dirs, files in os.walk(dest_folder):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_text = process_pdf(pdf_path)
                edgar_form_object = {
                        "companyName": company_name,
                        "content": pdf_text
                    }


                try:
                    client.data_object.create(edgar_form_object, "EdgarForm")
                    print(f'Added {file} to weaviate')
                except Exception as e:
                    print(f"Error saving {file} to Weaviate: {e}")


def download_file(url, dest_folder, company_name):
    headers = {"User-Agent": "Student Bot"}
    file_name = url.split("/")[-1]
    file_response = requests.get(url, headers=headers)
    time.sleep(1)
    with open(os.path.join(dest_folder, file_name), 'wb') as f:
        f.write(file_response.content)
    print(f"{file_name} downloaded.")
    if file_name.endswith('.htm'):
        pdf_path = os.path.join(dest_folder, file_name.replace('.htm', '.pdf'))
        pdfkit.from_file(os.path.join(dest_folder, file_name), pdf_path)
        print(f"{file_name} converted to pdf and saved.")

def download_forms(dest_folder, company_name):
    url = f'https://www.sec.gov/edgar/search/#/dateRange=all&category=form-cat1&entityName={quote(company_name)}'
    options = webdriver.ChromeOptions()
    options.add_argument("user-agent=Student Bot")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    current_page = 0
    def process_batch(url_batch):
        with ThreadPoolExecutor() as executor:
            executor.map(lambda url: download_file(url, dest_folder, company_name), url_batch)
        time.sleep(1)
    while True:
        print(f"Processing page {current_page + 1}...")
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'preview-file')))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        doc_page_links = [a['href'] for a in soup.select('.preview-file')]
        print(f"Found {len(doc_page_links)} document links on page {current_page + 1}.")
        if not doc_page_links:
            break
        cik_element = soup.find(attrs={"data-cik": True})
        cik = cik_element['data-cik'] if cik_element else '1318605'  # fallback to hardcoded CIK if not found
        doc_elements = soup.select('.preview-file')
        doc_info = [(a['data-adsh'], a['data-file-name']) for a in doc_elements]
        base_url = 'https://www.sec.gov/Archives/edgar/data/'
        urls = [f"{base_url}{cik}/{adsh.replace('-', '')}/{file_name}" for adsh, file_name in doc_info]
        for start in range(0, len(urls), 5):
            url_batch = urls[start:start + 5]
            process_batch(url_batch)
        current_page += 1
        next_page_url = f'{url}&start={current_page * 100}'
        driver.get(next_page_url)
    driver.quit()

def process_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_reader = PdfReader(f)
        pdf_text = ''
        for page_num in range(len(pdf_reader.pages)):  # corrected line
            pdf_text += pdf_reader.pages[page_num].extract_text()  # use the correct method to extract text
    return pdf_text

    
def main(company_name):
    destination_folder = f"/media/somebody/big_store/{company_name}_forms"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    download_forms(destination_folder, company_name)
    print(f"All {company_name} forms downloaded successfully.")
    save_to_weaviate(destination_folder, company_name)

if __name__ == '__main__':
    company_name = 'Coinbase Global'
    main(company_name)


    