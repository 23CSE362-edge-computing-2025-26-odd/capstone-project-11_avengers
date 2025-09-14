package com.yourproject.devices;

import java.util.*;

public class IoTDeviceCreator {
    
    // Simple replacement classes
    static class Application {
        String name;
        Map<String, AppModule> modules = new HashMap<>();
        List<AppEdge> edges = new ArrayList<>();
        Map<String, String> tupleMappings = new HashMap<>();

        Application(String name) { this.name = name; }

        void addAppModule(String name, int size) {
            modules.put(name, new AppModule(name, size));
        }

        void addAppEdge(String src, String dest, int latency, int bandwidth, String tupleType, String edgeType) {
            edges.add(new AppEdge(src, dest, latency, bandwidth, tupleType, edgeType));
        }

        void addTupleMapping(String module, String inputTuple, String outputTuple) {
            tupleMappings.put(inputTuple, outputTuple);
        }
    }

    static class AppModule {
        String name; int size;
        AppModule(String name, int size) { this.name = name; this.size = size; }
    }

    static class AppEdge {
        String src, dest, tupleType, edgeType;
        int latency, bandwidth;
        AppEdge(String src, String dest, int latency, int bandwidth, String tupleType, String edgeType) {
            this.src = src; this.dest = dest;
            this.latency = latency; this.bandwidth = bandwidth;
            this.tupleType = tupleType; this.edgeType = edgeType;
        }
    }

    static class Sensor {
        String id, type;
        Sensor(String id, String type) { this.id = id; this.type = type; }
    }

    static class Actuator {
        String id, type;
        Actuator(String id, String type) { this.id = id; this.type = type; }
    }

    // --- Logic equivalent to original ---
    public static void createIoTSensors(Application application) {
        Sensor heartRateSensor = new Sensor("heart_rate_sensor", "HR");
        Sensor bloodPressureSensor = new Sensor("bp_sensor", "BP");

        application.addAppEdge("HR", "data_fusion_module", 1000, 2000, "HR", "SENSOR");
        application.addAppEdge("BP", "data_fusion_module", 1000, 2000, "BP", "SENSOR");

        Actuator alertActuator = new Actuator("alert_actuator", "EMERGENCY_ALERT");
        application.addAppEdge("anomaly_detector_module", "EMERGENCY_ALERT", 100, 20, "ALERT", "ACTUATOR");
    }

    public static Application createApplication() {
        Application application = new Application("healthcare_app");
        application.addAppModule("data_fusion_module", 100);
        application.addAppModule("anomaly_detector_module", 100);
        application.addAppModule("task_dispatcher_module", 100);

        application.addTupleMapping("data_fusion_module", "HR_DATA", "FUSED_DATA");
        application.addTupleMapping("data_fusion_module", "BP_DATA", "FUSED_DATA");

        return application;
    }
}
