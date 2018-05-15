# -*- coding: utf-8 -*-
"""
Created on Tue May 08 18:51:30 2018

@author: Philipp
"""
import PyPDF2
#Note: another option would be textract, yet this does not even install when
#trying pip install textract. Might be worth a shot to try and fix this
#should the others not work
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

#Using PyPDF, does not include spaces, which makes it kind of useless
def extract_abstract_pypdf(file):
    #pdfFileObj = open(file)
    pdfReader = PyPDF2.PdfFileReader(file)
    print('Number of Pages', pdfReader.numPages)
    
    #This assumes, that the abstract is on the first page
    page1 = pdfReader.getPage(0).extractText()
    abstract_start = page1.find('Abstract')
    #workaround if headeing abstract is missing
    if (abstract_start == -1):
        abstract_start = 0
    print ('Abstract starts at', abstract_start)
    abstract_end = page1.find('Introduction')
    abstract = page1[abstract_start:abstract_end]
    print(abstract, '\n\n')
    
#Code found here, using pdfminer.six: 
#https://stackoverflow.com/questions/11087795/whitespace-gone-from-pdf-extraction-and-strange-word-interpretation
# convert pdf file to a string which has space among words 
def convert_pdf_to_txt(file):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'  # 'utf16','utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp  = open(file, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    text = retstr.getvalue()
    retstr.close()
    return text

def extract_abstract_pdfminer(file):
    text = convert_pdf_to_txt(file)
    
    abstract_start = text.find('Abstract')
    #workaround if heading abstract is missing
    if (abstract_start == -1):
        abstract_start = 0
    print ('Abstract starts at', abstract_start)
    abstract_end = text.find('Introduction')
    abstract = text[abstract_start:abstract_end]
    print(abstract, '\n\n')
    
extract_abstract_pypdf('./example1.pdf')
#Has an Abstract, but not the heading Abstract, so I cannot find it
#could resort to including everything up to introduction, as including 
#the title should not hurt the scores to much, but it's a workaround
extract_abstract_pypdf('./example2.pdf')

#Let's try the other version
extract_abstract_pdfminer('./example1.pdf')
extract_abstract_pdfminer('./example2.pdf')
#Works better, but takes longer