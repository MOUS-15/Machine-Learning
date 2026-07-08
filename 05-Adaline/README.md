# 🐾 Advanced Adaptive Linear Neuron (Adaline) Neural Network From Scratch

This repository features a highly optimized, fully documented, and object-oriented implementation of an **Adaptive Linear Neuron (Adaline)** network constructed entirely from scratch using `NumPy` and visualized via `Matplotlib`. The architecture automates binary classification, separating a synthesized multi-dimensional dataset of **Cats (Class -1)** and **Dogs (Class 1)** based on continuous clinical features ($X_1, X_2$).

---

## 🧠 1. البنية المعمارية والمفاهيم المتقدمة (Architectural & Advanced Concepts)

### 💡 The Theoretical Evolution: Adaline vs. Perceptron
* **The Perceptron Paradigm:** يعتمد نموذج البرسبترون التقليدي على دالة التنشيط القاطعة (Hard-limiting Step Function) لتحديث الأوزان. هذا يعني أن التحديث يحدث فقط إذا أخطأ النموذج في التصنيف النهائي للـ Discrete Labels. ينتج عن ذلك حساسية مفرطة للـ Noise، وتذبذب مستمر حول الحل، وعدم القدرة على الاستقرار التام إذا كانت البيانات تحتوي على أدنى تداخل.
* **The Adaline Paradigm:** يعتبر قفزة نوعية لأنه يطبق **قاعدة ويدرو-هوف (Widrow-Hoff Learning Rule)**، والتي تسمى أيضاً **قاعدة دلتا (Delta Rule)**. يتم تحديث الأوزان بناءً على **الناتج الخطي المستمر (Continuous Net Input)** قبل تمريره لدالة التنشيط القاطعة. هذا يحول عملية التعلم من مجرد محاولات عشوائية للتصنيف إلى مسألة تحسين رياضي (Mathematical Optimization Problem) تهدف لتقليل الخطأ التراكمي الكلي.



### 🚀 الميزات الهندسيّة المتقدمة في السكريبت (Advanced Functional Architecture)
1. **Vectorized Batch Gradient Descent (التهبيط التدريجي المصفوُفي الكلي):** الحسابات مُمكّنة بالكامل عبر المصفوفات (Fully Vectorized implementation). بدلاً من التحديث الفردي لكل عينة (Stochastic approach)، يقوم الكود بضرب المدخلات بالكامل وحساب تدرج دالة التكلفة (Gradient of Cost Function) لجميع العينات دفعة واحدة في كل دورة (Epoch)، مما يرفع كفاءة المعالجة الحسابية عبر الـ CPU.
2. **Mathematical Convergence Criteria & Early Stopping:** يمتلك النموذج آلية رصد متقدمة لإنهاء التدريب مبكراً وتوفير الموارد الحسابية. بدلاً من استهلاك الـ CPU في دورات غير لاهثة، يراقب النموذج التغير المطلق (Absolute Change) في دالة التكلفة بين الدورة الحالية والسابقة؛ فإذا كان الفارق أقل من $\epsilon = 10^{-5}$، يستنتج النموذج أنه وصل للقاع المستقر (Convergence Point) ويتوقف تلقائياً.
3. **Weight Initialization & Scale Strategy:** تم تهيئة مصفوفة الأوزان باستخدام التوزيع الطبيعي العشوائي (`np.random.randn`) مضروباً في مقياس صغير ($0.01$). هذه الاستراتيجية تحمي الشبكة من مشكلة **انفجار التدرج (Gradient Explosion)** التي تتسبب في تحول الأوزان إلى قيم غير معرفة (`NaN`) وتضمن استقرار الـ Convergence من اللحظة الأولى.

---

## 📐 2. الجزر الرياضي والاشتقاق التفصيلي (Mathematical Derivation & Formal Proof)

يعتمد الـ Adaline على تقليل دالة التكلفة المسماة **متوسط مربعات الأخطاء (Mean Squared Error - MSE)**، والتي تمثل سطحاً محدباً (Convex Surface) يشبه الوعاء، يمتلك نقطة قاع واحدة تسمى العالمية المثالية (Global Minimum)، مما يضمن عدم الوقوع في مصيدة القيعان المحلية (Local Minima).

### 1. معادلة التمرير الأمامي الخطي (Linear Combination)
لكل عينة داخل مصفوفة المدخلات $X$ التي تحتوي على الميزات وعمود الـ Bias، يُحسب المجموع الموزون كالتالي:
$$z = w_0x_1 + w_1x_2 + w_2b = \sum_{j=0}^{n} w_j x_j = X \cdot W$$

### 2. دالة التكلفة (Objective Cost Function)
يتم حساب التكلفة الكلية للنموذج عبر معادلة نصف متوسط مربعات الأخطاء (Sum of Squared Errors divided by 2m):
$$J(W) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - z^{(i)})^2$$
* حيث $m$ هو إجمالي عدد العينات (Total number of samples in matrix $X$).

### 3. اشتقاق قاعدة التحديث (Gradient Descent Derivation)
للانتقال نحو القاع، نحسب المشتقة الجزئية (Partial Derivative) لدالة التكلفة بالنسبة لكل وزن منفصل $w_j$:

$$\frac{\partial J}{\partial w_j} = \frac{\partial}{\partial w_j} \left[ \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - z^{(i)})^2 \right]$$

بتطبيق قاعدة السلسلة (Chain Rule) في التفاضل:
$$\frac{\partial J}{\partial w_j} = \frac{1}{2m} \sum_{i=1}^{m} 2(y^{(i)} - z^{(i)}) \cdot \frac{\partial}{\partial w_j} (y^{(i)} - z^{(i)})$$
$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - z^{(i)}) \cdot (-x_j^{(i)})$$
$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - z^{(i)}) x_j^{(i)}$$

بما أننا نتحرك **عكس اتجاه التدرج** لتصغير التكلفة (To minimize the cost function)، نقوم بطرح المشتقة مضروبة في معدل التعلم ($\eta$):
$$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial J}{\partial W}$$

بالتعويض الرياضي واختصار السوالب، نصل للمعادلة المصفوفية الفائقة المكتوبة داخل دالة الـ `fit`:
$$W_{\text{new}} = W_{\text{old}} + \frac{\eta}{m} \cdot X^T \cdot (y - z)$$

---

## 💻 3. التشريح الهيكلي للكود (Object-Oriented Architecture)

تم بناء الكود بالاعتماد على ممارسات الكود النظيف (Clean Code Standards) عبر تقسيم المهام كالتالي:

### 🛠️ Method Mechanics Breakdown:
* **`__init__(self, eta, epochs)`**:
  يقوم بتهيئة الـ Hyperparameters الأساسية للنموذج وضبط مؤشر العصر الأخير (`self.last_epoch`) لتوثيق دورة الاستقرار الحقيقية.
* **`prepare_data(self, number_of_samples)`**:
  مسؤول عن الـ Data Generation والتلاعب بالنطاقات العشوائية (Cats: 1-9, Dogs: 7-16) لخلق فضاء بيانات متوازن وقابل للفصل، مع دمج مصفوفات الفئات وضبط القيمة الرياضية للـ Bias إلى العمود الثابت $1$.
* **`fit(self)`**:
  قلب الخوارزمية النابض. يقوم بتنفيذ الـ Loop الكلي، وحساب مصفوفة الـ `error` المستمرة، وضخ التحديث للأوزان عبر ضرب مصفوفة الميزات المنقولة (`self.X.T`) في مصفوفة الأخطاء، مع توثيق الـ Cost التراكمي لكل دورة ومراقبة الـ Stop Criteria بدقة متناهية.
* **`predict(self, x1, x2)`**:
  محرك الاستدلال (Inference Engine). يستقبل مدخلات حية من المستخدم، يدمجها مع الـ Bias، ثم يطبق دالة التنشيط النهائية (Signum Function) لتحديد الهوية الطبقية للكائن (`Cat` أو `Dog`).
* **`accuracy(self)`**:
  مقياس الأداء الحسابي. يقوم بمسح المصفوفة الكلية للتحقق من نسبة دقة التنبؤات النهائية ومطابقتها التامة مع الـ Target Vector ومطابقتها باستخدام `np.where`.
* **`draw_data(self)`**:
  محلل الرؤية البيانية (Analytical Dashboard). يقوم برسم لوحتين (Subplots): اللوحة الأولى تفاعلية تظهر ميزات الكائنات وموقع **خط الفصل الرياضي المستنتج** من صفرية البرسبترون:
  $$x_2 = -\frac{w_0x_1 + w_2}{w_1}$$
  واللوحة الثانية تعرض منحنى هبوط التكلفة الاستقرائي الكلي للـ Adaline.
* **`print_info(self)`**:
  واجهة التقرير النهائي للـ Terminal. تقوم بعرض تقرير كامل مفرز بـ Dividers هندسية ليعرض للمستخدم قيم الأوزان المستقرة بدقة 4 أرقام عشرية، ونسبة الدقة المئوية الكلية.

---

## 🚀 4. دليل التشغيل والتحليل الفني (Execution & Expected Outputs)

### 📦 Installation & Dependencies
المشروع يعتمد بالكامل على البيئة الرياضية القياسية، تأكد من تنصيبها:
```bash
pip install numpy matplotlib
