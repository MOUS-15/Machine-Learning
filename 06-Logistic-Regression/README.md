# 🐾 Optimized Logistic Regression From Scratch (Binary Probabilistic Classifier)

This repository features a highly optimized, vectorized implementation of a **Logistic Regression Model** engineered entirely from scratch using `NumPy`. The project provides an advanced framework for binary classification, distinguishing between **Cats (Class 0)** and **Dogs (Class 1)** using custom synthesis tools and showcasing probability density mapping via interactive `Matplotlib` contour meshes.

---

## 🧠 1. البنية المعمارية والمفاهيم المتقدمة (Advanced Concepts & Architecture)

### 💡 The Quantum Leap: Logistic Regression vs. Adaline
* **The Adaline Architecture:** يعتمد على التنبؤ الخطي المباشر المستمر ($z = XW$). يتجاهل حقيقة أن فضاء التصنيف الثنائي محدد، مما يجعله يحسب تكلفة مشوهة للقيم البعيدة جداً عن خط القرار حتى لو كانت صحيحة.
* **The Logistic Regression Architecture:** يمرر الناتج الخطي $z$ عبر **دالة سيجمويد (Sigmoid Activation Function)**. تقوم هذه الدالة بضغط (Squeeze) النطاق اللانهائي للأرقام الحقيقية ليصبح محصوراً بدقة بين $0$ و $1$. هذا التحول يغير المفهوم من مجرد مسافة خطية إلى **احتمالية إحصائية حقيقية (True Statistical Probability)** تصف ثقة الموديل في القرار.



### 🚀 الميزات الهندسية للـ Implementation
1. **Probability Contour Surface Mesh (`plt.contourf`):** لا يكتفي الكود برسم خط قرار مصمت، بل يقوم بإنشاء شبكة إحداثيات متكاملة (Meshgrid) لحساب الاحتمالية عند كل نقطة في الفضاء ثنائي الأبعاد، ورسم تدرج لوني يعكس الانتقال التدريجي للاحتمالات (Confidence Gradient).
2. **Numerical Stability Strategy (`np.clip`):** لحماية الكود من الانهيار الرياضي القاتل الناتج عن حساب `log(0)` في دالة الـ Cost التراكمية، يطبق النموذج استراتيجية الـ Matrix Clipping بحصر قيم التنبؤات أوتوماتيكياً في النطاق الآمن $[10^{-15}, 1-10^{-15}]$ دون العبث بالقيم الأساسية لخط القرار.
3. **Hyperparameter Rigor & Vectorized Descent:** يتم التحديث والتحكم بالكامل بأسلوب مصفوفي دفعي (Batch Gradient Descent) يضمن الكفاءة العالية وسرعة الاستقرار الجبري.

---

## 📐 2. الجزر الرياضي والاشتقاق والـ Cost Function (Mathematical Foundations)

### 1. دالة التنشيط السيجمويدية (Sigmoid Function)
يتم تمرير الضرب القياسي للميزات والأوزان $z = X \cdot W$ عبر المعادلة التالية لإنتاج الاحتمالية $\hat{y}$:
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. دالة التكلفة (Binary Cross-Entropy Loss / Log Loss)
في الـ Logistic Regression، لا يمكن استخدام مربعات الأخطاء (MSE) لأن دالة التنشيط غير الخطية تحول السطح إلى سطح غير محدب مليء بالقيعان المحلية (Non-convex surface). البديل المثالي هو استخدام الـ Maximum Likelihood Estimation والتي ينتج عنها دالة الـ Cross-Entropy:
$$J(W) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 3. اشتقاق قاعدة التحديث (Mathematical Derivation)
لتقليل الـ Log Loss، نحسب المشتقة الجزئية بالنسبة للأوزان بتطبيق قاعدة السلسلة (Chain Rule)، مع العلم أن مشتقة السيجمويد هي $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:

$$\frac{\partial J}{\partial w_j} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_j}$$
$$\frac{\partial J}{\partial w_j} = \left[ -\frac{1}{m} \left( \frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}} \right) \right] \cdot \left[ \hat{y}(1-\hat{y}) \right] \cdot \left[ x_j \right]$$

بتبسيط المقادير الجبرية داخل الأقواس:
$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} \left( \frac{y(1-\hat{y}) - \hat{y}(1-y)}{\hat{y}(1-\hat{y})} \right) \cdot \hat{y}(1-\hat{y}) \cdot x_j$$
$$\frac{\partial J}{\partial w_j} = -\frac{1}{m} (y - y\hat{y} - \hat{y} + y\hat{y}) \cdot x_j = \frac{1}{m} (\hat{y} - y)x_j = -\frac{1}{m} (y - \hat{y})x_j$$

وبالتحرك عكس اتجاه التدرج لحساب الهبوط التدريجي للوزن، نصل للصيغة المصفوفية العبقرية المستخدمة في دالة الـ `fit`:
$$W_{\text{new}} = W_{\text{old}} + \frac{\eta}{m} \cdot X^T \cdot (y - \hat{y})$$

---

## 💻 3. معمارية البرمجة كائنية التوجه (OOP Architecture & Methods)

* **`prepare_data(number_of_samples)`**: توليد ميزات مستمرة للفئتين (Cats [0] & Dogs [1]) وحقن المصفوفة بعمود الـ Bias الافتراضي $1$.
* **`fit()`**: حلقة معالجة التدريب. تطبق التمرير الأمامي غير الخطي وحساب التباين، التحديث الممتد للأوزان، ورصد التوقف المبكر عند ثبات الـ Cost المطلق.
* **`predict(x1, x2)`**: محرك الاستنتاج الإحصائي. يحسب احتمالية النقطة ويصنفها بدقة بناءً على عتبة القرار (Decision Threshold = 0.5).
* **`accuracy()`**: فحص جودة التوقع وحساب النسبة المئوية للمطابقة التامة.
* **`draw_data()`**: لوحة القيادة التحليلية الرسومية. تعرض الـ Contourf Mesh لتدرج الاحتمالات متقاطعاً مع الـ Line الجبري الصفرى $x_2 = -\frac{w_0x_1 + w_2}{w_1}$ ومنحنى تناقص الـ Loss التراكمي.

---

## 🚀 4. دليل التشغيل الفني والمخرجات (Execution Guide)

### 📦 Dependencies
```bash
pip install numpy matplotlib
