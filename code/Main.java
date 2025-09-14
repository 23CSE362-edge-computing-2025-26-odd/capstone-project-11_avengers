import com.yourproject.devices.FogDeviceCreator;
import com.yourproject.devices.IoTDeviceCreator;
import com.yourproject.util.MetricsCollector;

import java.util.ArrayList;
import java.util.List;

public class Main {
    static List<FogDeviceCreator.FogDevice> fogDevices = new ArrayList<>();
    static List<IoTDeviceCreator.Sensor> sensors = new ArrayList<>();
    static List<IoTDeviceCreator.Actuator> actuators = new ArrayList<>();

    public static void main(String[] args) {
        System.out.println("=== Starting Simple IoT/Fog Simulation (without iFogSim) ===");

        // Step 1: Create Fog devices
        FogDeviceCreator.createFogDevices(fogDevices);
        System.out.println("\nCreated Fog Devices:");
        for (FogDeviceCreator.FogDevice fd : fogDevices) {
            System.out.println(fd);
        }

        // Step 2: Create Application (with modules, edges, and sensors/actuators)
        IoTDeviceCreator.Application application = IoTDeviceCreator.createApplication();
        IoTDeviceCreator.createIoTSensors(application);

        System.out.println("\nApplication Modules:");
        application.modules.values().forEach(m -> System.out.println(" - " + m.name));

        System.out.println("\nApplication Edges:");
        application.edges.forEach(e -> System.out.println(" - " + e.src + " -> " + e.dest +
                " [type=" + e.edgeType + ", tuple=" + e.tupleType + "]"));

        // Step 3: Demonstrate MetricsCollector
        MetricsCollector.Tuple t1 = new MetricsCollector.Tuple("TUPLE-1");
        MetricsCollector.recordLatency(t1, 15.2);
        MetricsCollector.recordMissedTask(t1);
        MetricsCollector.monitorResourceUtilization();

        System.out.println("\n=== Simulation finished! ===");
    }
}
