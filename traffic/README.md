# Experimentation process

- Start with a simple model to understand how changes affect the performance and avoid unnecessary complexity which only increases the calculation time.
- No proper accuracy with a without a rescaling layer or a convolutional layer. With one of both the test accuracy is around 90%.
- Significant jump to almost 97% accuracy when combining a rescaling layer with a covolutional layer.
- High filter kernel can make the results worse. While increasing the number of layers helps, it does not provide better results overall.
- Adding max pooling after the convolutional layer helped against overfitting increasing the accuracy while also reducing the calculation time.
- Adding dropout helps for overfitting increasing the accuracy by 0.5%.
- Adding a dense layer with lots of neurons helps for overfitting and increases the accuracy by another 0.5, but it also increases the calculation time as well.
- Finally adding another convolutional layer with max pooling right after the first one further increased the accuracy by almost 1 % to about 99% and almost equal performance on the training and test dataset.
