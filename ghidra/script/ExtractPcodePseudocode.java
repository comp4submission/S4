//Extract instructions, pcode, and pseudocode of functions.
//@author anonymous
//@category Basic script
//@keybinding
//@menupath
//@toolbar

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.AddressSetView;
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


public class ExtractPcodePseudocode extends GhidraScript {

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
        
        AddressSetView initialized = currentProgram.getMemory().getLoadedAndInitializedAddressSet();

        Map<String, List<String>> funcs_for_pcode = new HashMap<>();
        Map<String, String> funcs_for_pseudocode = new HashMap<>();

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

            List<String> pcodes = new ArrayList<>();

            for (Instruction inst : currentProgram.getListing().getInstructions(func.getBody(), true)) {
                // StringBuffer buffer = new StringBuffer();

                // buffer.append(inst.getMinAddress());
                // buffer.append(' ');
                // buffer.append(inst.getMnemonicString());
                // buffer.append(' ');

                // int nOperands = inst.getNumOperands();

                // for (int i = 0 ; i < nOperands ; i++) {
                //     String operand = inst.getDefaultOperandRepresentation(i);
                //     buffer.append(operand);
                //     buffer.append(' ');
                // }

                // // print instructions
                // println(buffer.toString());
                
                // extract p-code
                
                for (PcodeOp pcodeOp : inst.getPcode()) {
                    // println("PCode: " + pcodeOp);
                    pcodes.add(pcodeOp.getMnemonic());
                }
                funcs_for_pcode.put(func.getEntryPoint().toString(), pcodes);
            }
            // // print pseudocode
            String pseudocode = this.getDecompiledFunction(func);
            funcs_for_pseudocode.put(func.getEntryPoint().toString(), pseudocode);
        }

        // export the json
        String pcodeJson = new Gson().toJson(funcs_for_pcode);
        String pseudocodeJson = new Gson().toJson(funcs_for_pseudocode);

        // Get the program path
        String executablePath = currentProgram.getExecutablePath();
        String pcodeOutputPath = executablePath + ".pcode.json";
        String pseudocodeOutputPath = executablePath + ".pseudocode.json";

        // Output the pcode
        try {
            FileWriter writer = new FileWriter(pcodeOutputPath);
            writer.write(pcodeJson);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Output pseudocode
        try {
            FileWriter writer = new FileWriter(pseudocodeOutputPath);
            writer.write(pseudocodeJson);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    
    private String getDecompiledFunction(Function function) {
        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);

        DecompileOptions options = new DecompileOptions();
        options.setDefaultTimeout(30);// setting a 30 seconds timeout
        decompiler.setOptions(options);
        
        // de-compilation setting
        decompiler.toggleCCode(true);
        decompiler.toggleSyntaxTree(true);
        decompiler.setSimplificationStyle("decompile");


        DecompileResults results = decompiler.decompileFunction(function, decompiler.getOptions().getDefaultTimeout(), new ConsoleTaskMonitor());
        if (results == null || !results.decompileCompleted()) {
            println("Decompilation failed for function: " + function.getName());
            return "";
        }

        return results.getDecompiledFunction().getC();
    }
}
