# Java iFogSim Integration Guide

## Overview

This guide shows how to integrate the ECG Diagnosis model with iFogSim (Java-based fog computing simulation). The ECG model runs as a Python service on edge devices, while iFogSim simulates the fog computing infrastructure.

## Integration Options

### Option 1: HTTP REST API (Recommended)

Use the REST API wrapper to call the ECG model from Java.

#### 1. Start the Python API Server
```bash
# On the edge device/simulation host
cd CI/
python scripts/rest_api_wrapper.py
```

#### 2. Java iFogSim Integration Code

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.time.Duration;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

public class ECGDiagnosisModule {
    private static final String API_BASE_URL = "http://localhost:5000";
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    public ECGDiagnosisModule() {
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Diagnose ECG data and return results
     */
    public DiagnosisResult diagnoseECG(double[] ecgData, String deviceId) {
        try {
            // Create JSON request payload
            Map<String, Object> requestData = Map.of(
                "ecg_data", ecgData,
                "device_id", deviceId
            );
            
            String jsonRequest = objectMapper.writeValueAsString(requestData);
            
            // Create HTTP request
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(API_BASE_URL + "/diagnose"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonRequest))
                .timeout(Duration.ofSeconds(30))
                .build();
            
            // Send request and get response
            HttpResponse<String> response = httpClient.send(request, 
                HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() == 200) {
                JsonNode jsonResponse = objectMapper.readTree(response.body());
                return parseResponse(jsonResponse);
            } else {
                throw new RuntimeException("API call failed: " + response.statusCode());
            }
            
        } catch (Exception e) {
            System.err.println("ECG diagnosis failed: " + e.getMessage());
            return createErrorResult(e.getMessage());
        }
    }
    
    private DiagnosisResult parseResponse(JsonNode json) {
        return new DiagnosisResult(
            json.get("prediction").asText(),
            json.get("confidence").asDouble(),
            json.get("inference_time_ms").asDouble(),
            json.get("fuzzy_risk_score").asDouble(),
            json.get("fuzzy_risk_level").asText()
        );
    }
    
    private DiagnosisResult createErrorResult(String error) {
        return new DiagnosisResult("Error", 0.0, 0.0, 0.0, "Unknown");
    }
}

/**
 * Diagnosis result data class
 */
public class DiagnosisResult {
    public final String prediction;
    public final double confidence;
    public final double inferenceTimeMs;
    public final double riskScore;
    public final String riskLevel;
    
    public DiagnosisResult(String prediction, double confidence, 
                          double inferenceTimeMs, double riskScore, String riskLevel) {
        this.prediction = prediction;
        this.confidence = confidence;
        this.inferenceTimeMs = inferenceTimeMs;
        this.riskScore = riskScore;
        this.riskLevel = riskLevel;
    }
}
```

#### 3. Integration with iFogSim Application Module

```java
public class ECGProcessingModule extends AppModule {
    private ECGDiagnosisModule diagnosisModule;
    
    public ECGProcessingModule(String name, String appId, int userId, 
                              int mips, int ram) {
        super(name, appId, userId, mips, ram);
        this.diagnosisModule = new ECGDiagnosisModule();
    }
    
    @Override
    public void processEvent(SimEvent ev) {
        if (ev.getTag() == ECG_DATA_ARRIVED) {
            ECGDataTuple ecgTuple = (ECGDataTuple) ev.getData();
            
            // Record processing start time
            long startTime = System.currentTimeMillis();
            
            // Run ECG diagnosis
            DiagnosisResult result = diagnosisModule.diagnoseECG(
                ecgTuple.getEcgData(), 
                ecgTuple.getDeviceId()
            );
            
            long processingTime = System.currentTimeMillis() - startTime;
            
            // Create result tuple
            DiagnosisTuple diagnosisTuple = new DiagnosisTuple(
                ecgTuple.getDeviceId(),
                result,
                processingTime
            );
            
            // Send to next module or actuator
            sendUp(new FogEvent(DIAGNOSIS_COMPLETE, diagnosisTuple));
        }
        
        super.processEvent(ev);
    }
}
```

### Option 2: Process Execution (Alternative)

If REST API is not preferred, you can execute the Python script directly:

```java
public class ECGProcessExecutor {
    private static final String PYTHON_SCRIPT = "dist/ecg_edge_bundle_20250910_222617/edge_launcher.py";
    
    public DiagnosisResult diagnoseECG(String ecgFilePath) {
        try {
            // Create temporary output file
            String outputFile = "temp_result_" + System.currentTimeMillis() + ".json";
            
            // Execute Python script
            ProcessBuilder pb = new ProcessBuilder(
                "python", PYTHON_SCRIPT,
                "--input", ecgFilePath,
                "--output", outputFile
            );
            
            pb.directory(new File("CI/"));
            Process process = pb.start();
            
            // Wait for completion with timeout
            boolean finished = process.waitFor(30, TimeUnit.SECONDS);
            
            if (finished && process.exitValue() == 0) {
                // Read JSON result
                String jsonContent = Files.readString(Paths.get(outputFile));
                ObjectMapper mapper = new ObjectMapper();
                JsonNode result = mapper.readTree(jsonContent);
                
                // Clean up temp file
                Files.deleteIfExists(Paths.get(outputFile));
                
                return parseResponse(result);
            } else {
                throw new RuntimeException("Python process failed or timed out");
            }
            
        } catch (Exception e) {
            System.err.println("ECG diagnosis execution failed: " + e.getMessage());
            return createErrorResult(e.getMessage());
        }
    }
}
```

## iFogSim Simulation Integration

### 1. Define ECG Processing Application

```java
public class ECGDiagnosisApplication {
    public static Application createApplication(String appId, int userId) {
        Application application = Application.createApplication(appId, userId);
        
        // Define application modules
        application.addAppModule("ecg_sensor", 10);           // ECG data acquisition
        application.addAppModule("preprocessing", 100);       // Signal preprocessing  
        application.addAppModule("diagnosis", 500);           // ML inference
        application.addAppModule("risk_assessment", 50);      // Fuzzy logic
        application.addAppModule("alert_manager", 20);        // Alert generation
        
        // Define application edges (data flow)
        application.addAppEdge("ECG_SENSOR", "ecg_sensor", 1000, 500, "ECG_DATA", 
                              Tuple.UP, AppEdge.SENSOR);
        application.addAppEdge("ecg_sensor", "preprocessing", 1000, 500, "RAW_SIGNAL", 
                              Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("preprocessing", "diagnosis", 2000, 1000, "PROCESSED_SIGNAL", 
                              Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("diagnosis", "risk_assessment", 100, 100, "DIAGNOSIS_RESULT", 
                              Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("risk_assessment", "alert_manager", 100, 50, "RISK_ASSESSMENT", 
                              Tuple.UP, AppEdge.MODULE);
        application.addAppEdge("alert_manager", "ALERT_ACTUATOR", 100, 28, "ALERT", 
                              Tuple.DOWN, AppEdge.ACTUATOR);
        
        return application;
    }
}
```

### 2. Device Resource Configuration

```java
// Edge gateway with ECG processing capability
FogDevice ecgGateway = createFogDevice("ECG_Gateway", 
    2800, 4000,     // MIPS, RAM 
    10000,          // Uplink bandwidth
    10000,          // Downlink bandwidth  
    3,              // Level
    0.0107,         // Rate per MIPS
    16*103,         // Busy power
    16*83.25        // Idle power
);

// Add ECG processing capability
ecgGateway.setCanProcessECG(true);
ecgGateway.addProcessingModule(new ECGProcessingModule(
    "diagnosis", "ECGDiagnosisApp", userId, 500, 2048
));
```

### 3. Workload Generation

```java
public class ECGWorkloadGenerator {
    public static void generateECGWorkload(int deviceId, int userId, 
                                          String appId, double interval) {
        
        // Generate periodic ECG data
        int startTime = 0;
        int endTime = 10000; // 10 seconds simulation
        
        for (int time = startTime; time < endTime; time += interval) {
            // Generate ECG tuple
            ECGDataTuple ecgTuple = new ECGDataTuple(
                "ECG_SENSOR_" + deviceId,
                generateSyntheticECG(360 * 10), // 10 seconds at 360 Hz
                time
            );
            
            // Send to fog environment
            send(getId(), time, FogEvents.TUPLE_ARRIVAL, ecgTuple);
        }
    }
    
    private static double[] generateSyntheticECG(int samples) {
        // Generate synthetic ECG data for simulation
        double[] ecg = new double[samples];
        Random random = new Random();
        
        for (int i = 0; i < samples; i++) {
            // Simple synthetic ECG waveform
            double t = (double) i / 360; // Time in seconds
            ecg[i] = Math.sin(2 * Math.PI * 1.2 * t) * 0.5 + 
                    random.nextGaussian() * 0.1;
        }
        
        return ecg;
    }
}
```

## Performance Metrics

Track these metrics in your iFogSim simulation:

```java
public class ECGMetrics {
    public static class ProcessingMetrics {
        public double avgLatencyMs;
        public double maxLatencyMs; 
        public double throughputSamplesPerSec;
        public double accuracyRate;
        public double resourceUtilization;
        public int totalDiagnoses;
        public int errorCount;
    }
    
    public static void recordDiagnosis(DiagnosisResult result, long latency) {
        // Record metrics for analysis
        MetricsCollector.record("ecg.latency", latency);
        MetricsCollector.record("ecg.confidence", result.confidence);
        MetricsCollector.record("ecg.prediction", result.prediction);
        MetricsCollector.increment("ecg.total_diagnoses");
    }
}
```

## Deployment Instructions

### 1. For iFogSim developer:

```bash
# Get the ECG bundle
wget/curl the ecg_edge_bundle_20250910_222617.zip
# OR copy from your dist/ folder

# Extract bundle
unzip ecg_edge_bundle_20250910_222617.zip

# Install Python dependencies
cd ecg_edge_bundle_20250910_222617/
pip install -r requirements_minimal.txt

# Test the model
python edge_launcher.py --input data/sample_ecg.csv --output test_result.json

# Start REST API server (if using REST approach)
python ../rest_api_wrapper.py
```

### 2. Add Java dependencies to iFogSim project:

```xml
<!-- Add to pom.xml if using Maven -->
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-core</artifactId>
    <version>2.15.2</version>
</dependency>
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
    <version>2.15.2</version>
</dependency>
```

### 3. Integration Test:

```java
public class ECGIntegrationTest {
    public static void main(String[] args) {
        ECGDiagnosisModule ecgModule = new ECGDiagnosisModule();
        
        // Test with synthetic data
        double[] testData = generateTestECG();
        DiagnosisResult result = ecgModule.diagnoseECG(testData, "test_device");
        
        System.out.println("Diagnosis: " + result.prediction);
        System.out.println("Confidence: " + result.confidence);
        System.out.println("Latency: " + result.inferenceTimeMs + "ms");
    }
}
```
