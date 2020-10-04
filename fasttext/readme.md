fastText
github:https://github.com/facebookresearch/fastText

## vector
### values
```c++
std::vector<real> data_;
Vector::addRow
void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  A.addRowToVector(*this, i, a);
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  A.addRowToVector(*this, i);//*this表示的是Vector本身，比如hidden调用该addRow，则*this表示的就是hidden
}
```
## densematrix
### values
```c++
//在构造函数中初始化，为m*n的向量，m为词表大小，n为向量维度
std::vector<real> data_;
at
inline real& at(int64_t i, int64_t j) {
  return data_[i * n_ + j];//n_即n，向量维度
};
```
### DenseMatrix::addRowToVector
```c++
//此调用只需要修改Vector，hidden
void DenseMatrix::addRowToVector(Vector& x, int32_t i) const {
  assert(i >= 0);
  assert(i < this->size(0));
  assert(x.size() == this->size(1));
  for (int64_t j = 0; j < n_; j++) {
    x[j] += at(i, j);//at为内连函数
  }
}

//此调用修改data，为输入/出矩阵
void DenseMatrix::addVectorToRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}
```
## dictionary
### values
```c++
//最大词表大小，当词表大小达到最大值的0.75，就会进行一次word过滤
static const int32_t MAX_VOCAB_SIZE = 30000000;
static const int32_t MAX_LINE_SIZE = 1024; //默认读取一行，当一行word大于1024时，切分
std::shared_ptr<Args> args_;
std::vector<int32_t> word2int_;//在add和threshold中生成，下标为hash值，值为index

enum class entry_type : int8_t { word = 0, label = 1 };
struct entry {
  std::string word;
  int64_t count;
  entry_type type;
  std::vector<int32_t> subwords; // 子串，隐藏层取其向量的平均值
};
std::vector<entry> words_;//下标表示word的index
std::vector<real> pdiscard_;
Dictionary::initTableDiscard
void Dictionary::initTableDiscard() {
  pdiscard_.resize(size_);
  for (size_t i = 0; i < size_; i++) {
    real f = real(words_[i].count) / real(ntokens_);
    pdiscard_[i] = std::sqrt(args_->t / f) + args_->t / f;
  }
}
t=1e-4，f为频率，count为word出现的次数，ntokens_为总的word数。
这个pdiscard和word2vec论文中提到的有点不一样。
Dictionary::initNgrams
void Dictionary::initNgrams() {
  for (size_t i = 0; i < size_; i++) {
    std::string word = BOW + words_[i].word + EOW;
    words_[i].subwords.clear();
    words_[i].subwords.push_back(i);//将word放进subwords中
    if (words_[i].word != EOS) {
      computeSubwords(word, words_[i].subwords);//将子串放进subwords中
    }
  }
}
```
### Dictionary::add
```c++
void Dictionary::add(const std::string& w) {
  int32_t h = find(w);//hash查找
  ntokens_++;
  if (word2int_[h] == -1) {
    entry e;
    e.word = w;
    e.count = 1;
    e.type = getType(w);
    words_.push_back(e);
    word2int_[h] = size_++; //word index映射
  } else {
    words_[word2int_[h]].count++; //词表构建
  }
}
```
### Dictionary::readFromFile
```c++
void Dictionary::readFromFile(std::istream& in) {
  std::string word;
  int64_t minThreshold = 1;
  while (readWord(in, word)) {
    add(word);
    if (ntokens_ % 1000000 == 0 && args_->verbose > 1) {
      std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::flush;
    }
    if (size_ > 0.75 * MAX_VOCAB_SIZE) {
      minThreshold++;
      threshold(minThreshold, minThreshold);//当词表过大时删除，并重新构建index
    }
  }
  threshold(args_->minCount, args_->minCountLabel);//删除词，重新构建index
  initTableDiscard();//拒绝概率
  initNgrams();//初始化word的subwords，包含word本身和所有子串集合
  if (args_->verbose > 0) {
    std::cerr << "\rRead " << ntokens_ / 1000000 << "M words" << std::endl;
    std::cerr << "Number of words:  " << nwords_ << std::endl;
    std::cerr << "Number of labels: " << nlabels_ << std::endl;
  }
  if (size_ == 0) {
    throw std::invalid_argument(
        "Empty vocabulary. Try a smaller -minCount value.");
  }
}
```
### Dictionary::computeSubwords
该函数计算word中满足条件的子串，不包含word本身（按原参数的话如果word过短是会包含本身的，这个和word是否有冲突???）。
```c++
void Dictionary::computeSubwords(
    const std::string& word,
    std::vector<int32_t>& ngrams,
    std::vector<std::string>* substrings) const {
  for (size_t i = 0; i < word.size(); i++) {
    std::string ngram;
    if ((word[i] & 0xC0) == 0x80) {
      continue;
    }
    for (size_t j = i, n = 1; j < word.size() && n <= args_->maxn; n++) {
      ngram.push_back(word[j++]);
      while (j < word.size() && (word[j] & 0xC0) == 0x80) {
        ngram.push_back(word[j++]);
      }
      if (n >= args_->minn && !(n == 1 && (i == 0 || j == word.size()))) {
        int32_t h = hash(ngram) % args_->bucket;// 子串hash
        pushHash(ngrams, h);//hash index存入ngrams中
        if (substrings) {
          substrings->push_back(ngram);
        }
      }
    }
  }
}
```
"(word[i] & 0xC0) == 0x80"在附录中有提到，187行args_->bucket的值是2,000,000，maxn=6，minn=3。因为包含了<>两个字符，因此即便是一个字符其长度也等于3。
【186行&&后面那个条件没看懂???，args_->minn没理由等于1，就算等于1，满足该条件的只能是开始符和结束符，也就是<和>不会进词表，这样理解的话说得过去，因为如果1-gram的话，开始符和结束符增对fasttext模型确实没有含义】
### Dictionary::discard
```c++
bool Dictionary::discard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords_);
  if (args_->model == model_name::sup) {
    return false;
  }
  // pdiscard_[i]越小，被采样的概率越高，和频次成正比，不过频率不会小于10-4???
  return rand > pdiscard_[id];//rand是0~1之间产生的随机浮点数
}
```
### Dictionary::getLine
words会回传到line中，此为采样word，此中是有问题的。
对于每一行，对每个word进行拒绝采样，最终被保留到words中。
```c++
int32_t Dictionary::getLine(
    std::istream& in,
    std::vector<int32_t>& words,
    std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  std::string token;
  int32_t ntokens = 0;
  reset(in);
  words.clear();
  while (readWord(in, token)) {
    int32_t h = find(token);
    int32_t wid = word2int_[h];
    if (wid < 0) {
      continue;
    }
    ntokens++;
    if (getType(wid) == entry_type::word && !discard(wid, uniform(rng))) {
      words.push_back(wid);
    }
    if (ntokens > MAX_LINE_SIZE || token == EOS) {
      break;
    }
  }
  return ntokens;
}
```
## fasttext
### FastText::createRandomMatrix
```c++
std::shared_ptr<Matrix> FastText::createRandomMatrix() const {
  std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
      dict_->nwords() + args_->bucket, args_->dim);//注意输入矩阵的第一维是由nwords和bucket组成，dim默认值为100
  input->uniform(1.0 / args_->dim, args_->thread, args_->seed);//这个初始化还多线程
  return input;
}
```
### FastText::createTrainOutputMatrix
```c++
std::shared_ptr<Matrix> FastText::createTrainOutputMatrix() const {
  int64_t m =
      (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();//输出矩阵的第一维度，如果是分类就为label数，如果是词向量就为nwords
  std::shared_ptr<DenseMatrix> output =
      std::make_shared<DenseMatrix>(m, args_->dim);
  output->zero();
  return output;
}
```
### FastText::createLoss
按照skipgram来读的，因此关注negativesamplingloss
```c++
std::shared_ptr<Loss> FastText::createLoss(std::shared_ptr<Matrix>& output) {
  loss_name lossName = args_->loss;
  switch (lossName) {
    case loss_name::hs:
      return std::make_shared<HierarchicalSoftmaxLoss>(
          output, getTargetCounts());
    case loss_name::ns:
      return std::make_shared<NegativeSamplingLoss>(
          output, args_->neg, getTargetCounts());
    case loss_name::softmax:
      return std::make_shared<SoftmaxLoss>(output);
    case loss_name::ova:
      return std::make_shared<OneVsAllLoss>(output);
    default:
      throw std::runtime_error("Unknown loss");
  }
}
```
### FastText::skipgram
论文中提到过的窗口采样，ws默认是5，会先从[1,5]中均匀采样出boundary作为line[w]的窗口大小，其中的ngrams为line[w]的子串，对应的执行函数是Dictionary::computeSubwords，该ngrams并没有包含word本身。update中也没有提到word的embedding。
```c++
void FastText::skipgram(
    Model::State& state,
    real lr,
    const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  for (int32_t w = 0; w < line.size(); w++) {
    int32_t boundary = uniform(state.rng);
    const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()) {
        model_->update(ngrams, line, w + c, lr, state);
      }
    }
  }
}
```
### FastText::trainThread
```c++
void FastText::trainThread(int32_t threadId, const TrainCallback& callback) {
  std::ifstream ifs(args_->input);
  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);
  Model::State state(args_->dim, output_->size(0), threadId + args_->seed);
  const int64_t ntokens = dict_->ntokens();//语料全部token的个数，在add中计算
  int64_t localTokenCount = 0;
  std::vector<int32_t> line, labels;
  uint64_t callbackCounter = 0;
  try {
    while (keepTraining(ntokens)) {
      real progress = real(tokenCount_) / (args_->epoch * ntokens);
      if (callback && ((callbackCounter++ % 64) == 0)) {
        double wst;
        double lr;
        int64_t eta;
        std::tie<double, double, int64_t>(wst, lr, eta) =
            progressInfo(progress);
        callback(progress, loss_, wst, lr, eta);
      }
      real lr = args_->lr * (1.0 - progress);
      if (args_->model == model_name::sup) {
        localTokenCount += dict_->getLine(ifs, line, labels);
        supervised(state, lr, line, labels);
      } else if (args_->model == model_name::cbow) {
        localTokenCount += dict_->getLine(ifs, line, state.rng);
        cbow(state, lr, line);
      } else if (args_->model == model_name::sg) {
        localTokenCount += dict_->getLine(ifs, line, state.rng);
        skipgram(state, lr, line);
      }
      if (localTokenCount > args_->lrUpdateRate) {
        tokenCount_ += localTokenCount;
        localTokenCount = 0;
        if (threadId == 0 && args_->verbose > 1) {
          loss_ = state.getLoss();
        }
      }
    }
  } catch (DenseMatrix::EncounteredNaNError&) {
    trainException_ = std::current_exception();
  }
  if (threadId == 0)
    loss_ = state.getLoss();
  ifs.close();
}
```
### FastText::train
```c++
void FastText::train(const Args& args, const TrainCallback& callback) {
  args_ = std::make_shared<Args>(args);
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    throw std::invalid_argument("Cannot use stdin for training!");
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    throw std::invalid_argument(
        args_->input + " cannot be opened for training!");
  }
  dict_->readFromFile(ifs);//构建词表
  ifs.close();
  if (!args_->pretrainedVectors.empty()) {
    input_ = getInputMatrixFromFile(args_->pretrainedVectors);//可以提供前置训练
  } else {
    input_ = createRandomMatrix();//输入矩阵，为densematrix
  }
  output_ = createTrainOutputMatrix();//输出矩阵，为densematrix
  quant_ = false;
  auto loss = createLoss(output_);//loss函数选择，sk为negativesamplingloss
  bool normalizeGradient = (args_->model == model_name::sup);
  //注意model_,三个参数分别对应到Model中的wi，wo，loss
  model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
  startThreads(callback);
}
```
## model
### values
```c++
class Model {
 protected:
  std::shared_ptr<Matrix> wi_;//由wi初始化，构造函数中完成
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Loss> loss_;
  bool normalizeGradient_;
 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Loss> loss,
      bool normalizeGradient);
  Model(const Model& model) = delete;
  Model(Model&& model) = delete;
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;
  class State {
   private:
    real lossValue_;
    int64_t nexamples_;
   public:
    Vector hidden;
    Vector output;
    Vector grad;
    std::minstd_rand rng;
    State(int32_t hiddenSize, int32_t outputSize, int32_t seed);
    real getLoss() const;
    void incrementNExamples(real loss);
  };
...
```
### Model::computeHidden
此为求解子串向量的均值
```c++
void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);//wi_为输入矩阵，为densematrix
  }
  hidden.mul(1.0 / input.size());
}
```
### Model::State::incrementNExamples
```c++
void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}
Model::update
void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  computeHidden(input, state);//state.hidden已改变
  Vector& grad = state.grad;
  grad.zero();
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  state.incrementNExamples(lossValue);
  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, 1.0);//输入矩阵中subwords向量更新
  }
}
```
## loss
### BinaryLogisticLoss::binaryLogistic
```c++
real BinaryLogisticLoss::binaryLogistic(
    int32_t target,
    Model::State& state,
    bool labelIsPositive,
    real lr,
    bool backprop) const {
  real score = sigmoid(wo_->dotRow(state.hidden, target));
  if (backprop) {
    real alpha = lr * (real(labelIsPositive) - score);// g
    state.grad.addRow(*wo_, target, alpha);//grad不是0吗，应该是分类模型中被使用
    wo_->addVectorToRow(state.hidden, target, alpha); // 输出矩阵中target向量更新
  }
  if (labelIsPositive) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}
```
### NegativeSamplingLoss::forward
```c++
real NegativeSamplingLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];
  real loss = binaryLogistic(target, state, true, lr, backprop);
  for (int32_t n = 0; n < neg_; n++) {
    auto negativeTarget = getNegative(target, state.rng);
    loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
  }
  return loss;
}
```

## 疑问
1. 最大词表大小，当词表大小达到最大值的0.75，就会进行一次word过滤
2. discard采样函数和word2vec论文不一样
3. discard采样后生成的序列问题
## 附录
### C++相关
1. std::uniform_int_distribution
随机函数使用，一般先需要定义随机函数引擎，比如可以直接使用minstd_rand0 或 minstd_rand来定义，其产生的随机数的范围是[1,2147483646]之间，然后定义随机函数，最后使用。
在model.h中Model::State::rng定义了随机函数引擎：
std::minstd_rand rng; //定义引擎
Fasttext::skipgram中如下使用：
std::uniform_int_distribution<> uniform(1, args_->ws); //定义随机函数
int32_t boundary = uniform(state.rng); //使用
2. std::shared_ptr/std::make_shared
std::shared_ptr指向特定类型的对象，用于自动释放所指向的对象，一个最安全的分配和使用动态内存的方法是调用一个名为make_shared的函数。
std::make_shared在动态内存中分配一个对象并初始化它，返回指向此对象的shared_ptr
当要使用make_shared时，必须指定想要创建的对象，定义方式与模板类相同，在函数名之后跟一个尖括号，在其中给出类型。
std::shared_ptr<Model> model_;
model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
3. C++类的构造函数后面加:
构造函数后加单冒号的作用是初始化列表，对类成员变量初始化
比如下面：
声明
```c++
class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Loss> loss_;
  bool normalizeGradient_;
 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Loss> loss,
      bool normalizeGradient);
}
```
定义
```c++
Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}
```
4. C++文件流
fstream,ifstream,ostream
 ifstream -- 从已有的文件读
 ofstream -- 向文件写内容
 fstream - 打开文件供读写
对应的函数seekp:设置输出文件流指针位置，seekg：设置输入文件流指针位置。
### 一些知识点
1. (word[i] & 0xC0) == 0x80
0xc0对应二进制为11000000，word[i] & 0xC0即获取字节的前两位，0x80对应的二进制是10000000，==表示字节的前两位是否为10。而在utf编码中，字节的二进制表示中以10开头的字节都是多字节序列的后续字节。一个字节占8位，utf8编码中数字/英文占一个字节，汉字占3个到4个字节。


**阅读源码的方式：先大体的按其执行脚本过一遍，只需了解代码大体结构，函数作用即可，后面需要按代码结构反复阅读若干遍来弄懂细节。切勿一上来就把每个函数及其所有作用都搞懂。**