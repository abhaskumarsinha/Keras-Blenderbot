# Keras-Blenderbot
A Keras implementation of Blenderbot, a state-of-the-art open-domain chatbot model, supporting TensorFlow, PyTorch, and JAX backends seamlessly.

# Jump Directly into Colab to test? 
Go Ahead: [![image](https://github.com/abhaskumarsinha/Keras-Blenderbot/assets/31654395/3180ef9d-879c-43b6-880f-5ecfcb0da4be)](https://colab.research.google.com/github/abhaskumarsinha/Keras-Blenderbot/blob/main/Keras_Blenderbot.ipynb) (**Supports both - CPU and GPU**)


# ⚠️ WORK IN PROGRESS!

## Blenderbot Ported to Keras

Blenderbot-Keras is an experimental project that ports the 400M distilled version of the popular Chatbot model, Blenderbot, from PyTorch in HuggingFace to Keras v3. With Keras now supporting TensorFlow, PyTorch, and JAX backend seamlessly, this project aims to demonstrate the ease of transitioning models across frameworks without altering the codebase.

## Overview
Building upon the insights of prior research in open-domain chatbots, Blenderbot-Keras emphasizes the importance of various conversational skills for crafting engaging and natural interactions. Leveraging large-scale neural models and appropriate training data, Blenderbot-Keras strives to imbue chatbots with capabilities such as asking and answering questions, displaying knowledge, empathy, and personality.

## Features
- Ported 400M distilled version of Blenderbot to Keras v3
- Seamless support for both CPU and GPU without code modifications
- Includes a Colab file demonstrating:
  - Installation of the original PyTorch model
  - Building the model architecture in Keras
  - Transferring weights layer-by-layer
     Gradio UI for testing the model
- Experimentation with additional model support planned for the future
- Caching mechanism to speed up inference by 20x

## Abstract

 Blenderbot-Keras builds upon the foundation laid by previous work in the domain of chatbots, showcasing the efficacy of large-scale neural models and appropriate training strategies. By releasing variants of the Blenderbot model with varying parameter sizes and making both models and code publicly available, this project aims to advance the state-of-the-art in multi-turn dialogue systems. Human evaluations demonstrate the superiority of the best models in terms of engagingness and humanness measurements.

## Note
This project is currently at an experimental stage, and users are encouraged to contribute and provide feedback. Limitations and failure cases are under analysis to further refine the capabilities of the model.

**Disclaimer**: Blenderbot-Keras is not officially associated with the original Blenderbot project or HuggingFace. It is an independent effort aimed at exploring the capabilities of Keras for deploying state-of-the-art conversational AI models.
  

## References
1. Roller, Stephen, et al. "Recipes for building an open-domain chatbot." arXiv preprint arXiv:2004.13637 (2020).
