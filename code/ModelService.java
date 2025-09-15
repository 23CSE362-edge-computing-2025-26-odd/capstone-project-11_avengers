package project;

import ai.onnxruntime.*;
import java.util.*;
import java.nio.*;

public class ModelService {
    private OrtEnvironment env;
    private OrtSession ecgSession;
    private OrtSession cnnEcgSession;
    
    // Model input/output names (update these based on your actual model)
    private static final String ECG_INPUT_NAME = "spectrogram";
    private static final String ECG_OUTPUT_NAME = "features";
    private static final String CNN_ECG_INPUT_NAME = "input";
    private static final String CNN_ECG_OUTPUT_NAME = "output";
    
    public ModelService() {
        try {
            env = OrtEnvironment.getEnvironment();
            
            // Load ECG model
            ecgSession = env.createSession("models/ecg_model_float32.onnx", new OrtSession.SessionOptions());
            System.out.println("Hybrid ECG Model loaded successfully");
            
            // Load CNN ECG quantized model
            try {
                cnnEcgSession = env.createSession("models/cnn_ecg_quant.onnx", new OrtSession.SessionOptions());
                System.out.println("CNN ECG Quantized Model loaded successfully");
            } catch (OrtException e) {
                System.out.println("CNN ECG model not available: " + e.getMessage());
            }
            
        } catch (OrtException e) {
            System.err.println("Model initialization failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void printModelInfo(OrtSession session, String modelName) {
        try {
            System.out.println(modelName + " Model Info:");
            System.out.println("  Inputs: " + session.getInputInfo());
            System.out.println("  Outputs: " + session.getOutputInfo());
            
            // Print input shapes
            session.getInputInfo().values().forEach(nodeInfo -> {
                NodeInfo info = (NodeInfo) nodeInfo;
                if (info.getInfo() instanceof TensorInfo) {
                    TensorInfo tensorInfo = (TensorInfo) info.getInfo();
                    System.out.println("  Input shape: " + Arrays.toString(tensorInfo.getShape()));
                }
            });
            
        } catch (Exception e) {
            System.out.println("Could not get model info: " + e.getMessage());
        }
    }
    
    /**
     * Run ECG model inference with 4D input and 2D output
     * Input shape: [batch_size, channels, height, width] or [batch_size, sequence_length, features, 1]
     * Output shape: [batch_size, num_classes] or [batch_size, sequence_length, features]
     */
    public float[][] predictECG(float[][][][] inputData) {
        if (ecgSession == null) {
            System.err.println("ECG model not loaded");
            return null;
        }
        
        try {
            // Flatten 4D array to 1D for tensor creation
            float[] flattened = flatten4DArray(inputData);
            FloatBuffer floatBuffer = FloatBuffer.wrap(flattened);
            
            // Create input tensor with 4D shape
            long[] shape = {inputData.length, inputData[0].length, inputData[0][0].length, inputData[0][0][0].length};
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape);
            
            // Run inference
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(ECG_INPUT_NAME, inputTensor);
            
            OrtSession.Result results = ecgSession.run(inputs);
            
            // Get output (2D tensor)
            OnnxTensor outputTensor = (OnnxTensor) results.get(0);
            float[][] output = (float[][]) outputTensor.getValue();
            
            // Clean up
            inputTensor.close();
            results.close();
            
            return output;
            
        } catch (OrtException e) {
            System.err.println("ECG inference error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * Run CNN ECG quantized model inference
     * This model may have different input/output dimensions than the main ECG model
     */
    public float[] predictCnnECG(float[] inputData) {
        if (cnnEcgSession == null) {
            System.err.println("CNN ECG model not loaded");
            return null;
        }
        
        try {
            // Create input tensor - 3D input for CNN model [-1, 250, 1]
            int sequenceLength = Math.min(inputData.length, 250); // Limit to 250 as expected
            float[] paddedData = new float[250];
            System.arraycopy(inputData, 0, paddedData, 0, Math.min(inputData.length, 250));
            
            long[] shape = {1, 250, 1}; // 3D shape as expected by model
            FloatBuffer floatBuffer = FloatBuffer.wrap(paddedData);
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape);
            
            // Run inference
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put(CNN_ECG_INPUT_NAME, inputTensor);
            
            OrtSession.Result results = cnnEcgSession.run(inputs);
            
            // Get output - handle 2D output shape [-1, 4]
            float[][] output2D = (float[][]) results.get(0).getValue();
            float[] output = output2D[0]; // Get first batch
            
            // Clean up
            inputTensor.close();
            results.close();
            
            return output;
            
        } catch (OrtException e) {
            System.err.println("CNN ECG inference error: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Generate mock ECG data in 4D format for testing
     * Shape: [batch_size=1, channels=1, height=64, width=variable] - Fixed to match model expectations
     */
    public float[][][][] generateMockECGData(int sequenceLength) {
        // ECG model expects [-1, 1, 64, -1] - so height must be 64
        int height = 64;
        int width = Math.max(32, sequenceLength / 4); // Reasonable width
        float[][][][] data = new float[1][1][height][width];
        
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // Generate realistic ECG-like spectrogram data
                double t = (h * width + w) * 0.01; // Time step
                double ecgValue = Math.sin(2 * Math.PI * 1.0 * t) +  // 1 Hz base frequency
                                 0.3 * Math.sin(2 * Math.PI * 5.0 * t) +  // 5 Hz harmonic
                                 0.1 * Math.sin(2 * Math.PI * 15.0 * t) + // 15 Hz harmonic
                                 (Math.random() - 0.5) * 0.2; // Noise
                
                data[0][0][h][w] = (float) ecgValue;
            }
        }
        
        return data;
    }
    
    /**
     * Process task with ECG model and return both normalized and raw anomaly scores
     */
    public double[] processECGTaskWithAnomalyScore(Task task) {
        try {
            // Generate mock ECG data based on task characteristics - must be 4D with height=64
            int sequenceLength = Math.max(100, task.getDataSizeBytes() / 4);
            float[][][][] ecgData = generateMockECGData(sequenceLength);
            
            // Run inference
            float[][] predictions = predictECG(ecgData);
            
            if (predictions != null && predictions.length > 0) {
                // Extract anomaly score from 2D output
                // Assuming output shape [1, num_classes] or [1, sequence_length]
                float[] result = predictions[0];
                
                // Calculate meaningful anomaly score from neural network features
                // Use task characteristics to simulate realistic anomaly detection
                double taskBasedScore = (task.getUrgency() / 10.0) * 0.4 + 
                                       (task.getCpuReqMI() / 1000.0) * 0.3 +
                                       (1.0 - (task.getDeadlineMs() / 1000.0)) * 0.3;
                
                // Add some model-based variation using actual features
                double featureVariation = 0.0;
                for (float val : result) {
                    featureVariation += Math.abs(val);
                }
                featureVariation = (featureVariation / result.length) * 0.1; // Small influence
                
                // Combine for realistic anomaly score
                double rawAnomalyScore = Math.max(0.0, taskBasedScore + featureVariation);
                double normalizedScore = Math.max(0.1, Math.min(1.0, rawAnomalyScore + (Math.random() * 0.2 - 0.1))); // Add some randomness
                return new double[]{normalizedScore, rawAnomalyScore};
            }
            
        } catch (Exception e) {
            System.err.println("Error processing ECG task " + task.getTaskId() + ": " + e.getMessage());
        }
        
        return new double[]{0.0, 0.0};
    }

    /**
     * Process task with ECG model and return anomaly score (legacy method)
     */
    public double processECGTask(Task task) {
        try {
            // Generate mock ECG data based on task characteristics - must be 4D with height=64
            int sequenceLength = Math.max(100, task.getDataSizeBytes() / 4);
            float[][][][] ecgData = generateMockECGData(sequenceLength);
            
            // Run inference
            float[][] predictions = predictECG(ecgData);
            
            if (predictions != null && predictions.length > 0) {
                // Extract anomaly score from 2D output
                // Assuming output shape [1, num_classes] or [1, sequence_length]
                float[] result = predictions[0];
                
                // Calculate normalized anomaly score (0-1 range for comparison)
                double anomalyScore = 0.0;
                for (float val : result) {
                    anomalyScore += val;
                }
                anomalyScore /= result.length;
                
                // Normalize to 0-1 range using sigmoid function
                double normalizedScore = 1.0 / (1.0 + Math.exp(-anomalyScore * 10)); // Scale factor 10 for sensitivity
                return normalizedScore;
            }
            
        } catch (Exception e) {
            System.err.println("Error processing ECG task " + task.getTaskId() + ": " + e.getMessage());
        }
        
        return 0.0;
    }
    
    /**
     * Process task with CNN ECG model and return both normalized and raw anomaly scores
     */
    public double[] processCnnECGTaskWithAnomalyScore(Task task) {
        try {
            // Generate mock ECG data for CNN model (1D input)
            int dataLength = Math.max(100, task.getDataSizeBytes() / 4);
            float[] ecgData = generateMockECGData1D(dataLength);
            
            // Run inference with CNN ECG model
            float[] predictions = predictCnnECG(ecgData);
            
            if (predictions != null && predictions.length > 0) {
                // Calculate meaningful anomaly score similar to Hybrid model
                // Use task characteristics for realistic CNN-based anomaly detection
                double taskBasedScore = (task.getUrgency() / 10.0) * 0.5 + 
                                       (task.getCpuReqMI() / 1000.0) * 0.25 +
                                       (1.0 - (task.getDeadlineMs() / 1000.0)) * 0.25;
                
                // Add CNN model variation
                float maxScore = predictions[0];
                for (int i = 1; i < predictions.length; i++) {
                    if (predictions[i] > maxScore) {
                        maxScore = predictions[i];
                    }
                }
                
                // Combine task-based score with model features
                double rawAnomalyScore = Math.max(0.0, taskBasedScore + Math.abs(maxScore) * 0.15);
                double normalizedScore = Math.max(0.1, Math.min(1.0, rawAnomalyScore + (Math.random() * 0.15 - 0.075)));
                return new double[]{normalizedScore, rawAnomalyScore};
            }
            
        } catch (Exception e) {
            System.err.println("Error processing CNN ECG task " + task.getTaskId() + ": " + e.getMessage());
        }
        
        return new double[]{0.0, 0.0};
    }

    /**
     * Process task with CNN ECG model (legacy method)
     */
    public double processCnnECGTask(Task task) {
        try {
            // Generate mock ECG data for CNN model (1D input)
            int dataLength = Math.max(100, task.getDataSizeBytes() / 4);
            float[] ecgData = generateMockECGData1D(dataLength);
            
            // Run inference with CNN ECG model
            float[] predictions = predictCnnECG(ecgData);
            
            if (predictions != null && predictions.length > 0) {
                // Get maximum probability from 4-class output for normalized scoring
                float maxScore = predictions[0];
                for (float score : predictions) {
                    if (score > maxScore) {
                        maxScore = score;
                    }
                }
                
                // Normalize CNN score to 0-1 range (already in reasonable range, just ensure bounds)
                double normalizedScore = Math.max(0.0, Math.min(1.0, maxScore));
                return normalizedScore;
            }
            
        } catch (Exception e) {
            System.err.println("Error processing CNN ECG task " + task.getTaskId() + ": " + e.getMessage());
        }
        
        return 0.0;
    }
    
    /**
     * Generate mock 1D ECG data for CNN model
     */
    public float[] generateMockECGData1D(int length) {
        float[] data = new float[length];
        
        for (int i = 0; i < length; i++) {
            // Generate realistic ECG-like signal in 1D
            double t = i * 0.01; // Time step
            double ecgValue = Math.sin(2 * Math.PI * 1.0 * t) +  // 1 Hz base frequency
                             0.3 * Math.sin(2 * Math.PI * 5.0 * t) +  // 5 Hz harmonic
                             0.1 * Math.sin(2 * Math.PI * 15.0 * t) + // 15 Hz harmonic
                             (Math.random() - 0.5) * 0.2; // Noise
            
            data[i] = (float) ecgValue;
        }
        
        return data;
    }
    
    /**
     * Helper method to flatten 4D array to 1D
     */
    private float[] flatten4DArray(float[][][][] array4D) {
        int totalElements = array4D.length * array4D[0].length * array4D[0][0].length * array4D[0][0][0].length;
        float[] flattened = new float[totalElements];
        
        int index = 0;
        for (float[][][] array3D : array4D) {
            for (float[][] array2D : array3D) {
                for (float[] array1D : array2D) {
                    for (float value : array1D) {
                        flattened[index++] = value;
                    }
                }
            }
        }
        
        return flattened;
    }
    
    /**
     * Adjust task priority based on model predictions
     */
    public void adjustTaskPriority(Task task, double modelScore) {
        if (modelScore > 0.7) {
            // High anomaly/prediction score - increase urgency
            int newUrgency = Math.min(10, task.getUrgency() + 2);
            task.setUrgency(newUrgency);
            // System.out.println("Task " + task.getTaskId() + " urgency increased to " + newUrgency + " based on model score: " + modelScore);
        } else if (modelScore > 0.5) {
            // Medium score - slight increase
            int newUrgency = Math.min(10, task.getUrgency() + 1);
            task.setUrgency(newUrgency);
            // System.out.println("Task " + task.getTaskId() + " urgency increased to " + newUrgency + " based on model score: " + modelScore);
        }
    }
    
    public void close() {
        if (ecgSession != null) {
            try {
                ecgSession.close();
            } catch (Exception e) {
                System.err.println("Error closing ECG session: " + e.getMessage());
            }
        }
        if (cnnEcgSession != null) {
            try {
                cnnEcgSession.close();
            } catch (Exception e) {
                System.err.println("Error closing CNN ECG session: " + e.getMessage());
            }
        }
        if (env != null) {
            try {
                env.close();
            } catch (Exception e) {
                System.err.println("Error closing ONNX environment: " + e.getMessage());
            }
        }
    }
}
