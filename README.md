# mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations
This is the official repository for the [mini-vec2vec](https://www.arxiv.org/abs/2510.02348) paper. 
The method proposes a solution to the stability and cost problems of the [vec2vec](https://arxiv.org/abs/2505.12540) method for the unsupervised alignment of sentence embeddings.

## How to use?
#### Get the Requirements
Install the requirements (optionally inside a virtual environment):
```bash
pip install -r requirements.txt
```
#### Generate the Embeddings Yourself (Optional)
This part generates the embeddings we are going to align. It is provided here for completeness.

If you want to generate the embeddings yourself, you can run:
```bash
python generate_embeddings.py --output_dir output_dir/ --batch_size 32 --encoders e5,stella,granite,gtr
```
This will create one file per encoder in `output_dir/`. 

This step is not part of the alignment algorithm, of course, and can be safely ignored -- it just creates the data for the algorithm to run on. 

The output should be the same as vectors as in my huggingface repo [dar-tau/nq-embeddings](https://huggingface.co/dar-tau/nq-embeddings).

This step is more intensive than the alignment algorithm itself, taking approximately one hour on one T4 GPU. 


#### Run alignment 
Open the notebook in `linear_vec2vec.ipynb` and follow the instructions. You can change the encoders in:
```
embed_A, embed_B = sent_embeds['e5'].cpu(), sent_embeds['gtr'].cpu()
```

## How to Cite?
```
@misc{dar2025minivec2vec,
      title={mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations}, 
      author={Guy Dar},
      year={2025},
      eprint={2510.02348},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.02348}, 
}
```
