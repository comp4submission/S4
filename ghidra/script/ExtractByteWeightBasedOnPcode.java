//Extract instructions, pcode, and pseudocode of functions.
//@author anonymous
//@category Basic script
//@keybinding
//@menupath
//@toolbar

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.address.AddressSetView;
import ghidra.program.model.mem.Memory;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.pcode.PcodeOp;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.util.task.ConsoleTaskMonitor;
import ghidra.util.Msg;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;

import com.google.gson.Gson;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import java.nio.file.Path;
import java.nio.file.Paths;


public class ExtractByteWeightBasedOnPcode extends GhidraScript {

    @Override
    protected void run() throws Exception {
        currentProgram.setImageBase(toAddr(0), false);
        
        if (currentProgram == null) {
            Msg.info(this, "null current program: try again with the -process option");
            return;
        }

        if (currentProgram.getFunctionManager().getFunctionCount() == 0) {
            Msg.info(this, "No functions found in " + currentProgram.getName() + ", skipping.");
            return;
        }

        Memory memory = currentProgram.getMemory();
        
        AddressSetView initialized = currentProgram.getMemory().getLoadedAndInitializedAddressSet();

        Map<Long, Integer> byte_weight = new HashMap<>();

        for (Function func: currentProgram.getFunctionManager().getFunctions(true)) {
            // monitoring if user cancel the operation
            monitor.checkCancelled();
            // filter function
            if (func.isThunk()) {
                continue;
            }
            if (func.isExternal()) {
                continue;
            }
            if (!initialized.contains(func.getEntryPoint())) {
                continue;
            }
            if (currentProgram.getListing().getInstructionAt(func.getEntryPoint()) == null) {
                continue;
            }
            
            // print function name as well as the address
            String func_and_addr = func.getName() + "@" + func.getEntryPoint().toString();
            println(func_and_addr);

            for (Instruction inst : currentProgram.getListing().getInstructions(func.getBody(), true)) {

                Address address = inst.getAddress();

                // long instFileOffset = address.getOffset();

                long instFileOffset = memory.getAddressSourceInfo(address).getFileOffset();

                int instLength = inst.getLength();
                int pcodeNum = inst.getPcode().length;
                // println(inst.getMnemonicString() + " Length: " + instLength + "# of Pcode" + pcodeNum);

                for (int i = 0; i < instLength; i++) {
                    byte_weight.put(instFileOffset + i, pcodeNum);
                }
            }
        }

        // export the json
        String byteweightJson = new Gson().toJson(byte_weight);

        // Get the program path
        String executablePath = currentProgram.getExecutablePath();

        String byteWeightOutputPath = executablePath + ".byteweight.json";

        // Output byte weight
        try {
            FileWriter writer = new FileWriter(byteWeightOutputPath);
            writer.write(byteweightJson);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}
