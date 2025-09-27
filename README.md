# Ahmed Yahya AI Library

مكتبة خوارزميات الذكاء الاصطناعي من تطوير أحمد يحيى.
توفر هذه المكتبة مجموعة من الخوارزميات الأساسية في الذكاء الاصطناعي مثل الانحدار الخطي، والانحدار اللوجستي، وغيرها.

## الاعتمادات الرئيسية:
- numpy
- scikit-learn
- pandas

## الاستخدام:

```python
from ahmed_yahya_ai.algorithms import linear_regression

X = [1, 2, 3, 4]
y = [2, 4, 6, 8]
model = linear_regression(X, y)
print(model)
```

## التثبيت:
```bash
pip install .
```