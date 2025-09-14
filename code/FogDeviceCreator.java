

import java.util.*;

public class FogDeviceCreator {

    // Minimal replacement for FogDevice
    public static class FogDevice {
        private static int counter = 0;
        private final int id;
        private final String name;
        private final int mips;
        private final int ram;
        private final int upBw;
        private final int downBw;
        private int parentId = -1;

        public FogDevice(String name, int mips, int ram, int upBw, int downBw) {
            this.id = counter++;
            this.name = name;
            this.mips = mips;
            this.ram = ram;
            this.upBw = upBw;
            this.downBw = downBw;
        }

        public int getId() { return id; }
        public String getName() { return name; }
        public int getMips() { return mips; }
        public int getRam() { return ram; }
        public int getUpBw() { return upBw; }
        public int getDownBw() { return downBw; }
        public int getParentId() { return parentId; }

        public void setParentId(int parentId) { this.parentId = parentId; }

        @Override
        public String toString() {
            return "FogDevice{" +
                    "id=" + id +
                    ", name='" + name + '\'' +
                    ", mips=" + mips +
                    ", ram=" + ram +
                    ", upBw=" + upBw +
                    ", downBw=" + downBw +
                    ", parentId=" + parentId +
                    '}';
        }
    }

    // Create a simple hierarchy of fog devices
    public static void createFogDevices(List<FogDevice> fogDevices) {
        // Cloud
        FogDevice cloud = createFogDevice("Cloud", 20000, 16384, 100000, 100000);
        cloud.setParentId(-1);
        fogDevices.add(cloud);

        // Gateway
        FogDevice gateway = createFogDevice("Gateway", 10000, 1024, 10000, 10000);
        gateway.setParentId(cloud.getId());
        fogDevices.add(gateway);

        // Fog node
        FogDevice fogNode = createFogDevice("FogNode1", 2000, 512, 1000, 1000);
        fogNode.setParentId(gateway.getId());
        fogDevices.add(fogNode);

        // More devices could be added here
    }

    private static FogDevice createFogDevice(String name, int mips, int ram, int upBw, int downBw) {
        return new FogDevice(name, mips, ram, upBw, downBw);
    }
}
