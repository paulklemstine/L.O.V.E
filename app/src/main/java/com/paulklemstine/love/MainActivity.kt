package com.paulklemstine.love

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.text.Html
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import com.chaquo.python.PyException
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.mflisar.ansitohtml.AnsiToHtmlConverter
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.io.PrintStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var terminalTextView: TextView
    private lateinit var inputEditText: EditText
    private lateinit var scrollView: ScrollView

    private val executor = Executors.newSingleThreadExecutor()
    private val outputPipe = PipedOutputStream()
    private val inputPipe = PipedInputStream(outputPipe)

    private val inputQueue = PipedOutputStream()
    private val pythonInput = PipedInputStream(inputQueue)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        terminalTextView = findViewById(R.id.terminalTextView)
        inputEditText = findViewById(R.id.inputEditText)
        scrollView = findViewById(R.id.scrollView)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        System.setOut(PrintStream(outputPipe))
        System.setErr(PrintStream(outputPipe))
        System.setIn(pythonInput)

        inputEditText.setOnEditorActionListener { _, _, _ ->
            val inputText = inputEditText.text.toString() + "\n"
            inputEditText.text.clear()
            executor.execute {
                try {
                    inputQueue.write(inputText.toByteArray())
                    inputQueue.flush()
                } catch (e: Exception) {
                    // Ignore
                }
            }
            true
        }

        startPython()
        startOutputReader()
    }

    private fun startPython() {
        executor.execute {
            try {
                val py = Python.getInstance()
                val loveModule = py.getModule("love")
                loveModule.callAttr("main")
            } catch (e: PyException) {
                runOnUiThread {
                    terminalTextView.append("\nPython Error: ${e.message}")
                }
            }
        }
    }

    private fun startOutputReader() {
        executor.execute {
            val reader = inputPipe.bufferedReader()
            try {
                while (!executor.isShutdown) {
                    val line = reader.readLine()
                    if (line != null) {
                        runOnUiThread {
                            val html = AnsiToHtmlConverter.convert(line)
                            terminalTextView.append(Html.fromHtml(html, Html.FROM_HTML_MODE_LEGACY))
                            terminalTextView.append("\n")
                            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
                        }
                    } else {
                        break
                    }
                }
            } catch (e: Exception) {
                // Ignore
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdownNow()
        inputPipe.close()
        outputPipe.close()
        pythonInput.close()
        inputQueue.close()
    }
}
