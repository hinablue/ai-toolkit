import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function GET() {
  try {
    // Check if MPS is available
    const hasMps = await checkMps();

    if (!hasMps) {
      return NextResponse.json({
        hasMps: false,
        gpus: [],
        error: 'MPS not available on this system',
      });
    }

    // Get GPU stats
    const gpuStats = await getMpsGpuStats();

    return NextResponse.json({
      hasMps: true,
      gpus: gpuStats,
    });
  } catch (error) {
    console.error('Error fetching MPS GPU stats:', error);
    return NextResponse.json(
      {
        hasMps: false,
        gpus: [],
        error: `Failed to fetch GPU stats: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 },
    );
  }
}

async function checkMps(): Promise<boolean> {
  try {
    // Check if MPS is available using Python
    // This checks if torch.backends.mps.is_available() returns True
    const command = 'python3 -c "import torch; print(torch.backends.mps.is_available())"';
    const { stdout } = await execAsync(command);
    return stdout.trim() === 'True';
  } catch (error) {
    // If Python check fails, try to get GPU info from system_profiler as fallback
    try {
      await execAsync('system_profiler SPDisplaysDataType');
      return true;
    } catch {
      return false;
    }
  }
}

async function getMpsGpuStats() {
  try {
    // Get GPU information from system_profiler
    const { stdout } = await execAsync('system_profiler SPDisplaysDataType -json');
    const displayData = JSON.parse(stdout);

    const gpus: any[] = [];

    if (displayData.SPDisplaysDataType && Array.isArray(displayData.SPDisplaysDataType)) {
      displayData.SPDisplaysDataType.forEach((display: any, index: number) => {
        const gpuName = display._name || display.sppci_model || 'Unknown GPU';
        const vram = display.sppci_vram || display._spdisplays_vram || 'Unknown';

        // Parse VRAM (format: "8192 MB" or "8 GB")
        let memoryTotal = 0;
        if (typeof vram === 'string') {
          const vramMatch = vram.match(/(\d+)\s*(MB|GB)/i);
          if (vramMatch) {
            memoryTotal = parseInt(vramMatch[1]);
            if (vramMatch[2].toUpperCase() === 'GB') {
              memoryTotal *= 1024;
            }
          }
        }

        // For MPS, we don't have real-time monitoring data like nvidia-smi
        // So we'll use placeholder values or try to get basic info
        gpus.push({
          index: index,
          name: gpuName,
          driverVersion: 'MPS',
          temperature: 0, // MPS doesn't provide temperature monitoring
          utilization: {
            gpu: 0, // MPS doesn't provide real-time utilization
            memory: 0,
          },
          memory: {
            total: memoryTotal,
            free: memoryTotal, // We can't get real-time memory usage for MPS
            used: 0,
          },
          power: {
            draw: 0, // MPS doesn't provide power monitoring
            limit: 0,
          },
          clocks: {
            graphics: 0, // MPS doesn't provide clock monitoring
            memory: 0,
          },
          fan: {
            speed: 0, // MPS doesn't provide fan speed monitoring
          },
        });
      });
    }

    // If no GPUs found from system_profiler, create a default MPS entry
    if (gpus.length === 0) {
      gpus.push({
        index: 0,
        name: 'Apple GPU (MPS)',
        driverVersion: 'MPS',
        temperature: 0,
        utilization: {
          gpu: 0,
          memory: 0,
        },
        memory: {
          total: 0,
          free: 0,
          used: 0,
        },
        power: {
          draw: 0,
          limit: 0,
        },
        clocks: {
          graphics: 0,
          memory: 0,
        },
        fan: {
          speed: 0,
        },
      });
    }

    return gpus;
  } catch (error) {
    console.error('Error parsing GPU stats:', error);
    // Return a default MPS GPU entry if parsing fails
    return [
      {
        index: 0,
        name: 'Apple GPU (MPS)',
        driverVersion: 'MPS',
        temperature: 0,
        utilization: {
          gpu: 0,
          memory: 0,
        },
        memory: {
          total: 0,
          free: 0,
          used: 0,
        },
        power: {
          draw: 0,
          limit: 0,
        },
        clocks: {
          graphics: 0,
          memory: 0,
        },
        fan: {
          speed: 0,
        },
      },
    ];
  }
}
