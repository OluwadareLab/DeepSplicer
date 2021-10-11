 
# DeepSplicer: An Improved Method of Splice Sites Prediction using Deep Learning

**OluwadareLab,**
**University of Colorado, Colorado Springs**

----------------------------------------------------------------------
**Developers:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Victor Akpokiro<br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: vakpokir@uccs.edu <br /><br />

**Contact:** <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Oluwatosin Oluwadare, PhD <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Department of Computer Science <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;University of Colorado, Colorado Springs <br />
		 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Email: ooluwada@uccs.edu 
    
--------------------------------------------------------------------	

**1.	Content of folders:**
-----------------------------------------------------------	
* src: DeepSplicer source code. deepsplicer.py <br />
* src: Hyper-parameter tuning source code. <br />
* src: DeepSplicer cross-validation source code. deepsplicer_cross_val.py <br />
* Models file for deepsplicer models <br />
* Log file for utilization results logs <br />
* Plots file for utilization results plots <br />


**2.	Hi-C Data used in this study:**
-----------------------------------------------------------
In our study, we constructed genomic dataset from  [Adhikari, et al](https://pubmed.ncbi.nlm.nih.gov/32550561/). 
To demonstrate our model’s generality, we utilized five carefully selected datasets from organisms, namely: Homo sapiens, Oryza sativa japonica, Arabidopsis thaliana, Drosophila melanogaster, and Caenorhabditis elegans. We downloaded these reference genomic sequence datasets (FASTA file format) from [Albaradei, S. et al](https://pubmed.ncbi.nlm.nih.gov/32550561/) and its corresponding annotation sequence (GTF file format) from [Ensembl](https://uswest.ensembl.org/index.html). Our data for constructed to permit a 
**Sequence Length of 400** of 


**3.	One-Hot encoding:**
-----------------------------------------------------------

We used One-hot encoding to transforms our Genomic sequence data and labels into vectors of 0 and 1. In other words, each element in the vector will be 0, except the element that corresponds to the nucleotide base of the sequence data input is 1. Adenine (A) is [1 0 0 0], Cytosine (C) is [0 1 0 0], Guanine (G) is [0 0 1 0], Thymine (T) is [0 0 0 1].


**4.	Usage:**
----------------------------------------------------------- 
Usage: To use, type in the terminal python deepsplicer.py -n model_name -s sequence(acceptor or donor) -o organism_name -e encoded_sequnce_file -l encoded_label_file <br /> 	
                          		
                              
* **Arguments**: <br />	
	* name: A string for the name of the model <br />
	* sequence: A string to specify acceptor or donor input dataset<br />
	* organism: A string to specify organism name i.e ["hs", "at", "oriza", "d_mel", "c_elegans"] <br />
	* encoded sequence file: A file containing the encoded sequence data <br />
	* encoded label file: A file containing the encoded label data <br />



**6.	Output:**
-----------------------------------------------------------
Deepsplicer outputs three files: 

1. .h5: The deepslicer model and weight file.
2. .txt: A log file that contains the accuracy and evaluation metrics results.
3. png: contains the plotting of the prediction accuracy


**7.	Note:**
-----------------------------------------------------------
* Dataset sequence length is 400.
* Deepsplice folders [log, models, plots] is essential for code functionality.
* Genomic sequence input data should should transfomed using one-hot encoding.
