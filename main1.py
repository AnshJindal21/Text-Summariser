from pyscript import display
from pyscript import document
import networkx as nx

import math
import numpy as np
import pandas as pd
def mainfxn(event):
    input_data1 = document.querySelector("#DATA")
    data1 = input_data1.value
    
    input_model = document.querySelector("#model")
    model= input_model.value
    #display(model)
    if(model=="TF-IDF"):
        tfidf(data1)
    elif(model=="LSA"):
        lsa(data1)
    else :
        textrank(data1)
        
    
    output_div1 = document.querySelector("#output1")
    output_div1.innerText = data1
    output_div3 = document.querySelector("#output3")
    output_div3.innerText = model


def lsa(data1):
    display("\n\n")
    display("Your Summary is: \n")
    # Latent Semantic Analysis uses the following steps:
    # Step 1: Preprocessing of text data, tokenizing and all
    # Step 2: Formation of term document matrix, tf-idf vectorization 
    # Step 3: Applying Singular Value Decomposition to factorize tfidf vector
    # Step 4: Creatinng Summary

    # Install all the necessary python libraries
    #pip install numpy scikit-learn


    # Import all the necessary libraries
    import numpy as np
    import math
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import silhouette_score
    from collections import Counter



    # Preprocessing the text file obtained 
    # It returns the lower cased tokenized array 
    # Example:
    # If my sentence is "Hi, my name is Shubh Garg. I am currently doing my BTech from JIIT."
    # Then it will return [["hi my name is shubh garg"], [" i am currently doing my btech from jiit"]]
    # Space before 'i' in 2nd sentence as splitting on fullstop

    def is_alphanumeric(ch):
        # Function checks if character is either alphabet or number 
        return (ch>='a' and ch<='z') or (ch>='A' and ch<='Z') or (ch>='0' and ch<='9')

    def is_space_or_underscore(ch):
        # Function checks if character is a black space, underscore or not
        return ch == ' ' or ch == '_'

    def is_fullstop(ch):
        # Function checks if character is fullstop or not
        return ch == '.'

    def is_valid_char(ch):
        # Checks validity of character
        # Note: Change here if you want to get other types of characters in you paragraph
        # For example, if you want hyphen(-), then write a 'is_hyphen()' function and add ' or is_hyphen(ch)' in the return statement
        return is_alphanumeric(ch) or is_space_or_underscore(ch) or is_fullstop(ch)

    def is_upper_to_lower(ch):
        # Checks and convert upper case characters to lower case characters
        if (ch>='A' and ch<='Z'):
            ch = chr(ord(ch) + ord('a') - ord('A')) #ord is used to get ASCII value, and chr is used to convert from ASCII to char
        return ch

    def preprocess_text(text):
        # Convert everything to lower case and remove unnecessary characters
        lowercase_text = ""
        for ch in text:
            if is_valid_char(ch):
                lowercase_text += is_upper_to_lower(ch)

        # Split the array of sentences, based of fullstop
        split_sentences = lowercase_text.split('.')

        return split_sentences

    def tokenize_document(sentences):
        tokenized_documents = [sentence.lower().split() for sentence in sentences]

        # We are dropping single character words like 'a' or 'I' in the next code line as they don't give much value
        # So it return: [["hi", "my", "name", "is", "shubh garg"], ["am", "currently", "doing", "my", "btech", "from", "jiit"]]
        # It dropped 'i'
        tokenized_documents = [[word for word in sub_list if len(word) > 1] for sub_list in tokenized_documents]

        # Note: You also have to use stopwords here as they also don't add muchh value, as i did not have the file you are using, i could not use it

        return tokenized_documents



    def calculate_tf(word, document):
        # TF is calculated as (number of times word is present in the document)/(total number of words in the document)
        word_count = document.count(word)
        total_words = len(document)
        tf = word_count / total_words if total_words > 0 else 0
        return tf

    def calculate_idf(word, documents):
        # IDF is (total number of documents or sentences)/(number of documents in which word is present)
        num_documents_with_term = sum(1 for document in documents if word in document)
        total_documents = len(documents)
        # Adding 1 in division for edge case num_documents_with_term = 0
        # Adding 1 overall to deal with negative values of log
        # I refered GFG here, you can also search TFIDF calculation on GFG
        idf = 1 + math.log((total_documents + 1) / (1 + num_documents_with_term))
        return idf

    def create_term_document_matrix(tokenized_documents):
        # Get unique words
        unique_words = sorted(list(set(word for document in tokenized_documents for word in document)))

        # Calculate TF-IDF values
        term_document_matrix = []
        for document in tokenized_documents:
            tf_idf_values = [calculate_tf(word, document) * calculate_idf(word, tokenized_documents) for word in unique_words]
            term_document_matrix.append(tf_idf_values)
        return term_document_matrix




    # LSA on TDM is pretty standart process
    # Use SVD from Linear Algebra provided by numpy
    # It requires eigenvectors and eigenvalues, which, if you want to create the matrix, that again will use Linear Algebra to ease the work
    # Note: If you want User Defined SVD, contact me again
    def apply_lsa(term_document_matrix, k):
        # Perform Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(term_document_matrix, full_matrices=False)

        # Keep only the top k singular values and vectors
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        Vt_k = Vt[:k, :]

        # Compute the reduced term-document matrix
        reduced_term_document_matrix = np.dot(U_k, np.dot(S_k, Vt_k))

        return reduced_term_document_matrix




    def compute_sentence_scores(reduced_term_document_matrix):
        # Note: I am using a random scoring function i found online
        # It is dot multiplicaton, it is more similar to cosine similarity, if you want to search about it
        # Note: You have to change it with your own scoring function to find similarities
        # I couldn't understand your scoring function
        sentence_scores = np.dot(reduced_term_document_matrix, reduced_term_document_matrix.T).diagonal()
        return sentence_scores


    # The next two functions are doing the same work as your summary functions
    # Although I think the function you shared with me is incomplete, the basic work you want to do in that was to 
    # Select the sentences which have a good score, and then combining them
    def select_top_sentences(sentence_scores, num_sentences):
        # Sort the sentence scores in descending order
        sorted_sentence_indices = np.argsort(sentence_scores)[::-1]

        # Select the indices of the top num_sentences
        top_sentence_indices = sorted_sentence_indices[:num_sentences]
        return top_sentence_indices

    def generate_summary(sentences, top_sentence_indices):
        # Initialize the summary
        summary = ''

        # Concatenate the top sentences
        for i in top_sentence_indices:
            summary += sentences[i] + '. '

        return summary



        # Load the text
    
        # text = "Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, language understanding, and even decision-making. AI systems are designed to emulate human cognitive functions, utilizing algorithms and data to analyze patterns and make informed decisions. There are two main types of AI: narrow or weak AI, which is designed for a specific task, and general or strong AI, which aims to replicate human cognitive abilities across various domains. Machine learning is a subset of AI that involves training algorithms to recognize patterns and make predictions based on data. Natural Language Processing (NLP) is another essential component of AI, enabling machines to understand, interpret, and generate human-like language. AI applications are widespread and impact various industries, including healthcare, finance, transportation, and entertainment. The ethical implications of AI, such as bias in algorithms and potential job displacement, have also become important considerations. As AI continues to advance, researchers are exploring ways to ensure responsible and transparent development, addressing concerns about privacy, security, and the societal impact of these technologies. The quest for achieving artificial general intelligence, where machines can perform any intellectual task that a human can, remains a long-term goal in the field of AI."
    f=open("ansh2.txt","w")
    f.write(data1)
    f.close()   
    text = open("ansh2.txt","r").readline()
    # Print the text
    print("Text:")
    print(text)
    # Preprocess the text
    sentences = preprocess_text(text)
    # We have not tokenized the sentences here as we need 'sentences' in the later process
    # Instead, we have tokinized them in another function
    tokenized_document = tokenize_document(sentences)
    # Create the term-document matrix
    term_document_matrix = create_term_document_matrix(tokenized_document)
    # Apply LSA
    k = 10
    # k refers to top k singular values and vectors to obtain the reduced term-document matrix
    # You can change k
    reduced_term_document_matrix = apply_lsa(term_document_matrix, k)
    # Select the top sentences for the summary
    num_sentences = int(5)
    sentence_scores = compute_sentence_scores(reduced_term_document_matrix)
    top_sentence_indices = select_top_sentences(sentence_scores, num_sentences)
    summary = generate_summary(sentences, top_sentence_indices)
    # Print the summary
    display(summary)








def tokenise(g):
    temp=0
    list1=[]
    list3=[]
    while(temp==0):
        x=g.readline()
        if not x:
            break
        a=0
        for y in x.split():
            y=y.lower()
            if y[len(y)-1]=='.':
                list3.append(y)
                list1.append(list3)
                list3=[]
                continue
            list3.append(y)
        list1.append(list3)
        list3=[]
    return list1

N = ['stop', 'the', 'to', 'and', 'a', 'in', 'it', 'is', 'I', 'that', 'had', 'on', 'for', 'were', 'was']

def remove(list3):
    stop=[]
    temp=0
    for x in range(len(list3)):
        for y in range(len(list3[x])):
            if (list3[x][y][len(list3[x][y])-1]==',') or (list3[x][y][len(list3[x][y])-1]=='.'):
                if list3[x][y][:len(list3[x][y])-1] in N:
                    list3[x][y]="\0"
            else:
                if list3[x][y] in N:
                    list3[x][y]="\0"
    for x in range(len(list3)):
        while(list3[x].count("\0")):
            list3[x].remove("\0")
    return list3
def freq(list3):
    mydict={}
    for x in list3:
        for y in x:
            if y in mydict.keys():
                mydict[y]=mydict[y]+1
            else:
                mydict[y]=1
    return mydict
def score(list3,mydict):
    mydict2={}
    s=""
    list4=[]
    for x in list3:
        for y in x:
            s=s+y
            s=s+" "
        list4.append(s)
        s=""

    for x in range(len(list3)):
        mydict2[list4[x]]=0
        for y in list3[x]:
            mydict2[list4[x]]=mydict2[list4[x]]+mydict[y]
    return mydict2
def summary(a,sentence):
    mydict=sentence.copy()
    b=""
    while a>0:
        Keymax = max(zip(mydict.values(), mydict.keys()))[1]
        mydict[Keymax]=0
        b=b+Keymax
        a=a-1
    display(b)

def tfidf(data1):
    display("\n\n")
    display("Your Summary is: \n")
    f=open("ansh2.txt","w")
    f.write(data1)
    f.close()
    f=open("ansh2.txt","r")
    print(f.read())
    print("\n\n")
    f.close()
    f=open("ansh2.txt","r")
    list3=tokenise(f)
    list3=remove(list3)
    mydict=freq(list3)
    print(mydict)
    sentence=score(list3,mydict)
    print(sentence)
    a=int(3)
    summary(a,sentence)


def textrank(data1):
    display(" ")
    display(" ")
    display("Your Summary is: \n")
    display(" ")
    display(" ")
    f=open("ansh2.txt","w")
    f.write(data1)
    f.close()
    f=open("ansh2.txt","r")
    list3=tokenise(f)
    list3=remove(list3)
    if [] in list3:
        list3.remove([])
    dictlist=[] 
    for x in list3:
        thisdict={}
        for y in x:
            if y in thisdict:
                thisdict[y]+=1
            else:
                thisdict[y]=1
        dictlist.append(thisdict)
    rows, cols = (len(dictlist), len(dictlist))
    arr = [[1 for i in range(cols)] for j in range(rows)]
    for x in range(len(dictlist)):
        for y in range(len(dictlist)):
            f=0
            if(y==x):
                continue
            s1=0
            s2=0
            for a in dictlist[x].keys():
                s1+=dictlist[x][a]
            for b in dictlist[y].keys():
                s2+=dictlist[y][b]
            s1=math.sqrt(s1)
            s2=math.sqrt(s2)
            for a in dictlist[x].keys():
                for b in dictlist[y].keys():
                    if(a==b):
                        f+=dictlist[x][a]*dictlist[y][b]
            s1=s1*s2
            arr[x][y]=round(f/s1,3)
    print("  ",end="")
    for y in range(len(arr)):
            print(y+1,end=" ")
    print("\n")
    for x in range(len(arr)):
        print(x+1,end=" ")
        for y in range(len(arr)):
            print(arr[x][y],end=" ")
        print("\n")

    arr=np.array(arr)
    nx_graph = nx.from_numpy_array(arr)
    scores = nx.pagerank(nx_graph)
    # Extract top 10 sentences as the summary
    #ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(arr)), reverse=True)
    #for i in range(2):
    #  print(ranked_sentences[i][1])
    scores = sorted(scores.items(), key=lambda x:x[1],reverse=True)
    print(scores,"\n\n")
    a=int(5)
    b=""
    for i in range(a):
        for x in list3[scores[i][0]]:
            b=b+x+" "
    display(b)







#def textrank():
#    display("TEXT RANK CODE HERE")









