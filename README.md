# PyPanops
The project is a python implementation of the person clustering algorithm in the check-out free grocery vision system. Details of the algorithm are introduced in the paper [An Efficient Person Clustering Algorithm for Open Checkout-free Groceries](https://arxiv.org/abs/2208.02973). A large real-world dataset is released with project.

<p align="center"><img src="https://github.com/WuJunde/checkoutfree/blob/master/check.png" alt="text" width="1000"/></p>


## Dataset Download

https://drive.google.com/drive/folders/1gAw8SuVG82NWOlv77Pvt06Q7s5WNnwGI?usp=sharing

Two datasets are sequentially(aa -> ab -> ac ...) splited to 44 files and 10 files. Data is saved as json with the format {id:{'time':timestamp captured, 'ori': the orientation of person leaving the view, 'fea': features extracted by CNN, 'loc': the location the captured camera, 'label': the identification of captured person}}. Each piece of data represents a snapshot captured by a certain camera in the grocery. Each snapshot contains one and only one person. The snapshots are sorted by the captured time.

There are two datasets: DaiCOFG and IseCOFG, which collected from a large grocery and a small grocery respectively. DaiCOFG contains 362,300 snapshots with 10,176 identities for training, in which 125,378 snapshots are labeled, and 250,710 labeled snapshots with $7,406$ identities for testing. The snapshots are taken by 186 cameras deployed at the key spots of the grocery. IseCOFG contains 78,630 snapshots with 4,116 identities for training, in that 21,648 snapshots are labeled, and 54,606 snapshots with 2,773 people for testing. The snapshots are taken by 76 cameras in the grocery. 

## Quick Start

#### Training:

python main.py -mode train -data_path "input data path" -out_path 'output data path'

#### Test:

python main.py -mode test -data_path "input data path" -out_path 'output data path'

See cfg.py for more avaliable parameters

### Todo list

- [ ] GCN parallel processing & Buffer
- [x] del debug code
- [x] cls validation
- [ ] function name alignment
- [ ] del trials
- [ ] dataset preprocess tools
- [ ] nn optimization by toplist
- [ ] CSG & GCG optimization by sparse
