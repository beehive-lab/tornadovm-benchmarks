tornado --jvm="-Dtornado.device.memory=2GB -XX:+UnlockDiagnosticVMOptions -XX:+PrintAssembly" -cp target/tornadovm-benchmarks-1.0-SNAPSHOT.jar tornadovm.benchmarks.Main $@
