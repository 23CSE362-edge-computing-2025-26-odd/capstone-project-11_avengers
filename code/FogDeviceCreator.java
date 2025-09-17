package project;

import java.util.*;

public class FogDeviceCreator {

    public static class FogDevice {
        private static int counter = 0;
        private final int id;
        private final String name;
        private final int mips, ram, upBw, downBw;
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

    public static void createFogDevices(List<FogDevice> fogDevices) {
    	FogDevice cloud1 = createFogDevice("Cloud1", 20000, 16384, 100000, 100000);
    	cloud1.setParentId(-1);
    	fogDevices.add(cloud1);

    	FogDevice cloud2 = createFogDevice("Cloud2", 15000, 12000, 80000, 80000);
    	cloud2.setParentId(-1);
    	fogDevices.add(cloud2);


        FogDevice gateway = createFogDevice("Gateway", 10000, 1024, 10000, 10000);
        gateway.setParentId(cloud1.getId());
        fogDevices.add(gateway);
        
        

        // Multiple fog nodes under the gateway
        FogDevice fogNode1 = createFogDevice("FogNode1", 2000, 512, 1000, 1000);  // medium power
        fogNode1.setParentId(gateway.getId());
        fogDevices.add(fogNode1);

        FogDevice fogNode2 = createFogDevice("FogNode2", 3000, 1024, 1500, 1500); // stronger
        fogNode2.setParentId(gateway.getId());
        fogDevices.add(fogNode2);

        FogDevice fogNode3 = createFogDevice("FogNode3", 1500, 256, 800, 800);    // weaker
        fogNode3.setParentId(gateway.getId());
        fogDevices.add(fogNode3);
    }


    private static FogDevice createFogDevice(String name, int mips, int ram, int upBw, int downBw) {
        return new FogDevice(name, mips, ram, upBw, downBw);
    }
}
