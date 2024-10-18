'use client'

import { useState, useEffect } from 'react'
import { Upload, Loader2, Leaf, AlertCircle } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import axios from 'axios'

const generateLLMResponse = async (prediction: string): Promise<string> => {
  // Replace this with the actual Gemini Pro API call
  const response = await axios.post('/api/gemini-pro', {
    prompt: `Generate a response for the plant disease prediction: ${prediction}`,
  })
  return response.data.text
}

// SQL function to store image and prediction in the database
const saveToDatabase = async (imageUrl: string, prediction: string) => {
  // Send the image URL and prediction to your backend API that stores it in SQL
  try {
    await axios.post('/api/save-image-prediction', { imageUrl, prediction })
  } catch (error) {
    console.error('Error saving to the database:', error)
  }
}

export default function PlantDiseasePredictorApp() {
  const [model, setModel] = useState<tf.GraphModel | null>(null)
  const [image, setImage] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [llmResponse, setLlmResponse] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadModel = async () => {
      try {
        // Replace 'model_url' with the actual URL or path to your TensorFlow.js model
        const loadedModel = await tf.loadGraphModel('model_url')
        setModel(loadedModel)
      } catch (error) {
        console.error('Error loading model:', error)
        setError('Failed to load the prediction model. Please try again later.')
      }
    }
    loadModel()
  }, [])

  const preprocessImage = async (file: File): Promise<tf.Tensor> => {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224]) // Adjust size based on your model's input requirements
          .toFloat()
          .expandDims()
        resolve(tensor)
      }
      img.onerror = (error) => reject(error)
      img.src = URL.createObjectURL(file)
    })
  }

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && model) {
      const imageUrl = URL.createObjectURL(file)
      setImage(imageUrl)
      setIsLoading(true)
      setPrediction(null)
      setLlmResponse(null)
      setError(null)
      try {
        const tensor = await preprocessImage(file)
        const predictions = await model.predict(tensor) as tf.Tensor
        const topPrediction = Array.from(await predictions.data())
          .map((p, i) => ({ probability: p, className: i }))
          .sort((a, b) => b.probability - a.probability)[0]
        
        // Replace with your actual class names
        const classNames = ['Healthy', 'Powdery Mildew', 'Rust', 'Scab']
        const predictedClass = classNames[topPrediction.className]
        setPrediction(predictedClass)

        // Save the image and prediction to the database
        await saveToDatabase(imageUrl, predictedClass)

        const llmResult = await generateLLMResponse(predictedClass)
        setLlmResponse(llmResult)
      } catch (error) {
        console.error('Error analyzing image:', error)
        setError('An error occurred while analyzing the image. Please try again.')
      } finally {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="min-h-screen h-screen w-screen flex items-center justify-center bg-gradient-to-br from-gray-950 to-black p-4 overflow-auto">
      <Card className="w-full max-w-4xl bg-black/80 backdrop-blur-md shadow-2xl rounded-3xl overflow-hidden border border-emerald-500/20">
        <CardHeader className="bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 text-black p-8 relative overflow-hidden">
          <div className="absolute inset-0 bg-black/10 backdrop-blur-sm"></div>
          <div className="relative z-10">
            <CardTitle className="text-4xl font-bold text-center mb-3">Plant Disease Predictor</CardTitle>
            <CardDescription className="text-center text-emerald-900 text-lg font-semibold">
              Upload an image of a plant to get an AI-powered disease prediction and treatment advice
            </CardDescription>
          </div>
          <Leaf className="absolute top-6 right-6 h-16 w-16 text-emerald-900/30" />
        </CardHeader>
        <CardContent className="p-8 space-y-8">
          <div className="flex justify-center">
            <label htmlFor="image-upload" className="cursor-pointer group">
              <div className="w-80 h-80 rounded-3xl bg-gradient-to-br from-emerald-600 to-cyan-600 p-1 flex items-center justify-center overflow-hidden shadow-inner transition-all duration-300 ease-in-out group-hover:shadow-lg group-hover:scale-105">
                <div className="w-full h-full rounded-3xl bg-gray-950 flex items-center justify-center overflow-hidden">
                  {image ? (
                    <img src={image} alt="Uploaded plant" className="w-full h-full object-cover" />
                  ) : (
                    <div className="text-center p-8 bg-gray-900/70 backdrop-blur-sm rounded-2xl shadow-lg transition-all duration-300 group-hover:bg-gray-800/80 group-hover:shadow-xl">
                      <Upload className="mx-auto h-20 w-20 text-emerald-400 mb-4 transition-transform group-hover:scale-110 duration-300" />
                      <p className="text-xl font-medium text-emerald-300">Click to upload an image</p>
                      <p className="mt-2 text-sm text-emerald-400/70">or drag and drop</p>
                    </div>
                  )}
                </div>
              </div>
              <input
                id="image-upload"
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </label>
          </div>
          {isLoading && (
            <div className="text-center p-6 bg-gray-900/70 backdrop-blur-sm rounded-2xl shadow-lg">
              <Loader2 className="h-12 w-12 animate-spin text-emerald-400 mx-auto mb-3" />
              <p className="text-xl font-medium text-emerald-300">Analyzing image...</p>
            </div>
          )}
          {error && (
            <Alert variant="destructive" className="bg-red-950/50 border-red-500 text-red-200">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {prediction && (
            <div className="bg-gray-900/80 backdrop-blur-sm p-8 rounded-2xl shadow-lg transition-all duration-300 hover:shadow-xl hover:bg-gray-900/90 border border-emerald-500/20">
              <h3 className="text-2xl font-semibold text-emerald-400 mb-4">AI Prediction:</h3>
              <p className="text-emerald-200 text-lg mb-4">The plant appears to be affected by: <strong className="text-emerald-300">{prediction}</strong></p>
              {llmResponse && (
                <div>
                  <h4 className="text-xl font-semibold text-emerald-400 mb-2">Detailed Analysis:</h4>
                  <p className="text-emerald-200 whitespace-pre-line leading-relaxed text-lg">{llmResponse}</p>
                </div>
              )}
            </div>
          )}
          <div className="text-center">
            <Button
              onClick={() => document.getElementById('image-upload')?.click()}
              className="bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-black font-medium py-3 px-8 rounded-full shadow-lg transition-all duration-300 hover:shadow-xl transform hover:-translate-y-1 text-lg"
            >
              Upload New Image
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}