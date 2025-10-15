package com.jules.borg.launcher

import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.jules.borg.launcher.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val termuxPackage = "com.termux"
    private val termuxRunCommand = "com.termux.app.RunCommandService"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.launchButton.setOnClickListener {
            if (isPackageInstalled(termuxPackage)) {
                launchTermuxBootstrap()
            } else {
                promptInstallTermux()
            }
        }
    }

    private fun isPackageInstalled(packageName: String): Boolean {
        return try {
            packageManager.getPackageInfo(packageName, 0)
            true
        } catch (e: PackageManager.NameNotFoundException) {
            false
        }
    }

    private fun launchTermuxBootstrap() {
        val bootstrapCommand = """
            pkg install -y git && \
            if [ -d "${'$'}{HOME}/Borg" ]; then
                cd "${'$'}{HOME}/Borg" && git pull;
            else
                git clone https://github.com/paulklemstine/Borg.git "${'$'}{HOME}/Borg";
            fi && \
            bash "${'$'}{HOME}/Borg/termux_bootstrap.sh"
        """.trimIndent()

        val intent = Intent().apply {
            setClassName(termuxPackage, termuxRunCommand)
            action = "com.termux.RUN_COMMAND"
            putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash")
            putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-c", bootstrapCommand))
            putExtra("com.termux.RUN_COMMAND_WORKDIR", "/data/data/com.termux/files/home")
            putExtra("com.termux.RUN_COMMAND_BACKGROUND", false)
        }

        try {
            startService(intent)
            // Also bring Termux to the foreground
            val launchIntent = packageManager.getLaunchIntentForPackage(termuxPackage)
            if (launchIntent != null) {
                startActivity(launchIntent)
            }
            finish() // Close the launcher app
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to launch Termux. Is it installed correctly?", Toast.LENGTH_LONG).show()
        }
    }

    private fun promptInstallTermux() {
        Toast.makeText(this, "Termux is required. Redirecting to F-Droid for installation.", Toast.LENGTH_LONG).show()
        val intent = Intent(Intent.ACTION_VIEW).apply {
            data = Uri.parse("https://f-droid.org/en/packages/com.termux/")
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        try {
            startActivity(intent)
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(this, "Could not open browser to F-Droid.", Toast.LENGTH_SHORT).show()
        }
    }
}